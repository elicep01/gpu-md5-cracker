#include "md5_cpu.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <atomic>
#include <chrono>

using namespace std::chrono;

/* ================= minimal MD5 (public-domain) ================= */

static inline uint32_t rotl(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

void md5_raw(const unsigned char* msg, size_t len, unsigned char d[16]) {
    unsigned char blk[64] = {0};
    std::memcpy(blk, msg, len);
    blk[len] = 0x80;
    uint64_t bits = len * 8ULL;
    std::memcpy(blk + 56, &bits, 8);
    uint32_t a = 0x67452301, b = 0xefcdab89,
             c = 0x98badcfe, d0 = 0x10325476;
    const uint32_t* X = reinterpret_cast<const uint32_t*>(blk);
    const uint32_t K[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
    };
    const int s[64] = {
        7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
        5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
        4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
        6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
    };
    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if      (i < 16) F = (b & c) | (~b & d0), g = i;
        else if (i < 32) F = (d0 & b) | (~d0 & c), g = (5*i + 1) & 15;
        else if (i < 48) F = b ^ c ^ d0,          g = (3*i + 5) & 15;
        else             F = c ^ (b | ~d0),       g = (7*i) & 15;
        F += a + K[i] + X[g];
        a = d0; d0 = c; c = b; b += rotl(F, s[i]);
    }
    a += 0x67452301; b += 0xefcdab89; c += 0x98badcfe; d0 += 0x10325476;
    std::memcpy(d,   &a, 4);
    std::memcpy(d+4, &b, 4);
    std::memcpy(d+8, &c, 4);
    std::memcpy(d+12,&d0,4);
}

std::string to_hex(const unsigned char* d) {
    char buf[33];
    for (int i = 0; i < 16; ++i) std::sprintf(&buf[2*i], "%02x", d[i]);
    buf[32] = '\0';
    return buf;
}

std::string md5_hash(const char* s) {
    unsigned char d[16];
    md5_raw(reinterpret_cast<const unsigned char*>(s), std::strlen(s), d);
    return to_hex(d);
}

inline void idx_to_pw(uint64_t idx, char pw[PASSWORD_LEN+1]) {
    for (int p = PASSWORD_LEN - 1; p >= 0; --p) {
        pw[p] = CHARSET[idx % CHARSET_SIZE];
        idx  /= CHARSET_SIZE;
    }
    pw[PASSWORD_LEN] = '\0';
}

double brute_force_md5(const std::string& target,
                       bool& found_out, std::string& result,
                       int threads)
{
    int64_t total = 1;
    for (int i = 0; i < PASSWORD_LEN; ++i) total *= CHARSET_SIZE;

    std::atomic_bool found(false);
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        uint64_t chunkSize = total / nthreads;
        uint64_t start = tid * chunkSize;
        uint64_t end = (tid == nthreads - 1) ? total : start + chunkSize;

        char pw[PASSWORD_LEN+1];
        for (uint64_t idx = start; idx < end; ++idx) {
            if (found.load(std::memory_order_relaxed)) break;
            idx_to_pw(idx, pw);
            if (md5_hash(pw) == target) {
                if (!found.exchange(true, std::memory_order_relaxed)) {
                    result = pw;
                }
                break;
            }
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

    int thr = omp_get_max_threads();
    if (argc == 4 && std::string(argv[2]) == "-t")
        thr = std::stoi(argv[3]);

    std::string tgt(argv[1]);
    std::transform(tgt.begin(), tgt.end(), tgt.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    bool found;
    std::string pw;

    high_resolution_clock::time_point start1 = high_resolution_clock::now();
    double sec = brute_force_md5(tgt, found, pw, thr);
    high_resolution_clock::time_point end = high_resolution_clock::now();

    duration<double> chrono_sec = duration_cast<duration<double>>(end - start1);

    if (found) std::cout << "CPU found: " << pw << "\n";
    else       std::cout << "CPU NOT found\n";

    std::cout << thr << "," << chrono_sec.count() << std::endl;
    return 0;
}
