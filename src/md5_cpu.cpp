/*  md5_cpu.cpp – CPU-side brute-force MD5 cracker
 *  ------------------------------------------------
 *  – runtime-selectable password length (≤ MAX_PW_LEN)
 *  – OpenMP parallel search with early-exit
 *  – exposes cpu_crack() helper for the unified CLI
 */
#include "md5_cpu.h"      // CHARSET, CHARSET_SIZE, prototypes
#include "cli_config.h"   // MAX_PW_LEN, extern int g_pw_len

#include <omp.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <atomic>
#include <cstdio>
#include <cstdint>

/* ------------------------------------------------------------------ */
/*  Global runtime knob – defaults to the maximum but can be lowered  */
/*  from cli_driver.cpp simply by writing g_pw_len = <1‒7>.           */
int g_pw_len = MAX_PW_LEN;
/* ------------------------------------------------------------------ */


/* ========================= minimal MD5 ============================ */
/* Public-domain single-block MD5 implementation (512-bit message)   */

static inline uint32_t rotl(uint32_t x, uint32_t c)
{ return (x << c) | (x >> (32 - c)); }

void md5_raw(const unsigned char* msg, size_t len, unsigned char d[16])
{
    unsigned char blk[64] = {0};
    std::memcpy(blk, msg, len);
    blk[len] = 0x80;                         // 1-bit terminator
    uint64_t bits = len * 8ULL;
    std::memcpy(blk + 56, &bits, 8);         // length footer

    uint32_t a = 0x67452301, b = 0xefcdab89,
             c = 0x98badcfe, d0 = 0x10325476;

    const uint32_t* X = reinterpret_cast<const uint32_t*>(blk);
    static const uint32_t K[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391 };
    static const int s[64] = {
         7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
         5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
         4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
         6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21 };

    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if      (i < 16) F = (b & c) | (~b & d0), g = i;
        else if (i < 32) F = (d0 & b) | (~d0 & c), g = (5*i + 1) & 15;
        else if (i < 48) F = b ^ c ^ d0,           g = (3*i + 5) & 15;
        else             F = c ^ (b | ~d0),        g = (7*i) & 15;

        F += a + K[i] + X[g];
        a  = d0; d0 = c; c = b; b += rotl(F, s[i]);
    }

    a += 0x67452301; b += 0xefcdab89;
    c += 0x98badcfe; d0 += 0x10325476;

    std::memcpy(d   , &a, 4);
    std::memcpy(d+4 , &b, 4);
    std::memcpy(d+8 , &c, 4);
    std::memcpy(d+12, &d0,4);
}

/* -------- hex helpers ------------------------------------- */

std::string to_hex(const unsigned char* d)
{
    char buf[33];
    for (int i = 0; i < 16; ++i) std::sprintf(&buf[2*i], "%02x", d[i]);
    buf[32] = '\0';
    return buf;
}

std::string md5_hash(const char* s)
{
    unsigned char d[16];
    md5_raw(reinterpret_cast<const unsigned char*>(s),
            std::strlen(s), d);
    return to_hex(d);
}

/* -------- idx → password ---------------------------------- */

inline void idx_to_pw(uint64_t idx, int len, char* pw)
{
    for (int p = len - 1; p >= 0; --p) {
        pw[p] = CHARSET[idx % CHARSET_SIZE];
        idx  /= CHARSET_SIZE;
    }
    pw[len] = '\0';
}

/* -------- parallel brute-force ----------------------------- */

double brute_force_md5(const std::string& target,
                       bool& found_out, std::string& result,
                       int threads, int len)
{
    /* total = CHARSET_SIZE ^ len   (len ≤ 7, so fits in int64) */
    int64_t total = 1;
    for (int i = 0; i < len; ++i) total *= CHARSET_SIZE;

    std::atomic_bool found(false);
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic,64) num_threads(threads)
    for (int64_t idx = 0; idx < total; ++idx) {
        if (found.load(std::memory_order_relaxed)) continue;

        char pw_buf[MAX_PW_LEN + 1];
        idx_to_pw(static_cast<uint64_t>(idx), len, pw_buf);

        if (md5_hash(pw_buf) == target) {
            if (!found.exchange(true, std::memory_order_relaxed))
                result = pw_buf;      // only once
        }
    }

    found_out = found.load();
    return omp_get_wtime() - t0;
}

/* -------- thin wrapper for the unified CLI ---------------- */

int cpu_crack(const std::string& digest, int threads)
{
    if (threads <= 0) threads = omp_get_max_threads();

    bool   found;
    std::string pw;
    double sec = brute_force_md5(digest, found, pw, threads, g_pw_len);

    if (found) std::cout << "[CPU] " << pw << "  (" << sec << " s)\n";
    else       std::cout << "[CPU] NOT FOUND (" << sec << " s)\n";

    return found ? 0 : 2;
}
