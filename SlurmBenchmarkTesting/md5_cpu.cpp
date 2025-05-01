#include "md5_cpu.h"
#include <omp.h>
#include <iostream>
#include <cstring>
#include <atomic>

// 1) Move MD5 constants out of the function so they're not re-allocated on each call
static const uint32_t K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};
static const int s[64] = {
    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
    5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
};

// 2) Predefine your charset once
#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_SIZE (sizeof(CHARSET)-1)

// 3) Rotate-left helper marked inline
static inline uint32_t rotl(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

// 4) md5_raw marked inline to encourage in-lining
inline void md5_raw(const unsigned char* msg, size_t len, unsigned char d[16]) {
    unsigned char blk[64] = {0};
    std::memcpy(blk, msg, len);
    blk[len] = 0x80;
    uint64_t bits = len * 8ULL;
    std::memcpy(blk + 56, &bits, 8);

    uint32_t a=0x67452301, b=0xefcdab89,
             c=0x98badcfe, d0=0x10325476;
    const uint32_t* X = reinterpret_cast<const uint32_t*>(blk);

    // 5) Unroll loop manually for a few iterations can help, but here rely on compiler
    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if      (i < 16) F = (b & c) | (~b & d0), g = i;
        else if (i < 32) F = (d0 & b) | (~d0 & c), g = (5*i + 1) & 15;
        else if (i < 48) F = b ^ c ^ d0,          g = (3*i + 5) & 15;
        else             F = c ^ (b | ~d0),       g = (7*i) & 15;

        F += a + K[i] + X[g];
        a = d0; d0 = c; c = b; b += rotl(F, s[i]);
    }

    a += 0x67452301; b += 0xefcdab89;
    c += 0x98badcfe; d0 += 0x10325476;
    std::memcpy(d,   &a, 4);
    std::memcpy(d+4, &b, 4);
    std::memcpy(d+8, &c, 4);
    std::memcpy(d+12,&d0,4);
}

// 6) Convert hex target string to raw bytes once, before parallel region
static void hex2bytes(const char* hex, unsigned char out[16]) {
    auto cv = [&](char c)->int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    for (int i = 0; i < 16; ++i)
        out[i] = (cv(hex[2*i]) << 4) | cv(hex[2*i+1]);
}

// 7) idx→password helper unrolled slightly for speed
inline void idx_to_pw(uint64_t idx, char pw[PASSWORD_LEN+1]) {
    for (int p = PASSWORD_LEN - 1; p >= 0; --p) {
        pw[p] = CHARSET[idx % CHARSET_SIZE];
        idx /= CHARSET_SIZE;
    }
    pw[PASSWORD_LEN] = '\0';
}

double brute_force_md5(const unsigned char target_raw[16],
                       bool& found_out, std::string& result,
                       int threads)
{
    // 8) Compute total dynamically
    uint64_t total = 1;
    for (int i = 0; i < PASSWORD_LEN; ++i) total *= CHARSET_SIZE;

    // 9) Atomic flag aligned to cache line to avoid false sharing
    alignas(64) std::atomic<bool> found(false);

    double t0 = omp_get_wtime();

    // 10) Use static scheduling with a chunk tuned to your system
    #pragma omp parallel for schedule(static, 1024) num_threads(threads)
    for (uint64_t idx = 0; idx < total; ++idx) {
        // Early exit
        if (found.load(std::memory_order_relaxed)) continue;

        char pw[PASSWORD_LEN+1];
        idx_to_pw(idx, pw);

        unsigned char hash[16];
        md5_raw(reinterpret_cast<const unsigned char*>(pw), PASSWORD_LEN, hash);

        // 11) Compare raw bytes directly
        bool match = true;
        for (int i = 0; i < 16; ++i) {
            if (hash[i] != target_raw[i]) { match = false; break; }
        }
        if (match) {
            if (!found.exchange(true, std::memory_order_relaxed))
                result = pw;
        }
    }

    found_out = found.load();
    return omp_get_wtime() - t0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::puts("Usage: cpu_crack <32-char MD5> [-t N]");
        return 1;
    }

    int threads = omp_get_max_threads();
    if (argc == 4 && std::string(argv[2]) == "-t")
        threads = std::stoi(argv[3]);

    // 12) Prepare raw target bytes
    unsigned char target_raw[16];
    hex2bytes(argv[1], target_raw);

    bool found;
    std::string pw;
    double elapsed = brute_force_md5(target_raw, found, pw, threads);

    if (found) std::cout << "CPU found: " << pw << "\n";
    else       std::cout << "CPU NOT found\n";
    std::cout << "CPU time (" << threads << " threads): " << elapsed << " s\n";
    return 0;
}