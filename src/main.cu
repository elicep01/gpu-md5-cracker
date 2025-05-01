/********************************************************************
*  main.cu  –  GPU MD5 brute-forcer (runtime password length ≤ 7)
*  ------------------------------------------------------------------
*  – length set at run time via g_pw_len (host) → d_pw_len (device)
*  – exposes  extern "C" int gpu_crack(const std::string&)  for CLI
********************************************************************/

#include "md5_cpu.h"      // CHARSET, CHARSET_SIZE
#include "cli_config.h"   // MAX_PW_LEN, g_pw_len
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <iostream>

/* ---------- charset + constants --------------------------------- */
#define HOST_CHARSET_SIZE CHARSET_SIZE          // 62

__device__ __constant__
char d_CHARSET[HOST_CHARSET_SIZE + 1] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/* MD5 “K” table in constant memory */
static const uint32_t h_K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391 };

__device__ __constant__ uint32_t d_K[64];

/* runtime length, copied from host before launch */
__device__ __constant__ int d_pw_len;

/* ---------- global early-exit flags (managed) ------------------- */
__device__ __managed__ volatile int      g_found = 0;
__device__ __managed__ uint64_t          g_idx   = 0;

/* ---------- helpers (device) ------------------------------------ */

__device__ __forceinline__
uint32_t rotl(uint32_t x, uint32_t c) { return (x << c) | (x >> (32 - c)); }

/* Minimal single-block MD5 (≤ 7 chars, len = d_pw_len) */
__device__ void md5_single(const char* in, unsigned char dig[16])
{
    unsigned char blk[64] = {0};
    /* copy message + terminator */
    for (int i = 0; i < d_pw_len; ++i) blk[i] = in[i];
    blk[d_pw_len] = 0x80;
    uint64_t bits = static_cast<uint64_t>(d_pw_len) * 8ULL;
    *reinterpret_cast<uint64_t*>(blk + 56) = bits;

    const uint32_t* X = reinterpret_cast<uint32_t*>(blk);

    uint32_t a = 0x67452301, b = 0xefcdab89,
             c = 0x98badcfe, d = 0x10325476;

    const int s[64] = {
        7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
        5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
        4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
        6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21 };

    #pragma unroll 64
    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if      (i < 16) { F = (b & c) | (~b & d);        g = i; }
        else if (i < 32) { F = (d & b) | (~d & c);        g = (5*i + 1) & 15; }
        else if (i < 48) { F = b ^ c ^ d;                 g = (3*i + 5) & 15; }
        else             { F = c ^ (b | ~d);              g = (7*i) & 15; }

        F += a + d_K[i] + X[g];
        a  = d; d = c; c = b; b += rotl(F, s[i]);
    }

    a += 0x67452301; b += 0xefcdab89;
    c += 0x98badcfe; d += 0x10325476;
    uint32_t regs[4] = { a, b, c, d };

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dig[4*i  ] =  regs[i]        & 0xFF;
        dig[4*i+1] = (regs[i] >>  8) & 0xFF;
        dig[4*i+2] = (regs[i] >> 16) & 0xFF;
        dig[4*i+3] = (regs[i] >> 24) & 0xFF;
    }
}

/* ---------- brute-force kernel ---------------------------------- */
__global__ __launch_bounds__(256,4)
void brute_kernel(const unsigned char* target, uint64_t total_pw)
{
    if (g_found) return;

    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t idx    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    char pw_buf[MAX_PW_LEN + 1];
    unsigned char dig[16];

    for ( ; idx < total_pw; idx += stride) {
        if (g_found) return;

        uint64_t tmp = idx;
        /* idx → password */
        for (int p = d_pw_len - 1; p >= 0; --p) {
            pw_buf[p] = d_CHARSET[tmp % HOST_CHARSET_SIZE];
            tmp /= HOST_CHARSET_SIZE;
        }
        pw_buf[d_pw_len] = '\0';

        md5_single(pw_buf, dig);

        const uint4* d4 = reinterpret_cast<const uint4*>(dig);
        const uint4* t4 = reinterpret_cast<const uint4*>(target);

        if (d4->x == t4->x && d4->y == t4->y &&
            d4->z == t4->z && d4->w == t4->w) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0)
                g_idx = idx;
            return;
        }
    }
}

/* ---------- host helpers ---------------------------------------- */
static inline uint64_t pow64(int base, int exp)
{
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i) r *= base;
    return r;
}

static inline unsigned char hex2byte(char hi, char lo)
{
    auto v = [](char c)->int{
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    return (v(hi) << 4) | v(lo);
}

/* idx → password  (host, for final print) */
static void idx_to_pw(uint64_t idx, int len, char* out)
{
    for (int p = len - 1; p >= 0; --p) {
        out[p] = CHARSET[idx % HOST_CHARSET_SIZE];
        idx   /= HOST_CHARSET_SIZE;
    }
    out[len] = '\0';
}

/* ---------- public driver (called by CLI) ----------------------- */
extern "C" int gpu_crack(const std::string& digest)
{
    /* --- preprocess target hash --- */
    unsigned char h_target[16];
    for (int i = 0; i < 16; ++i)
        h_target[i] = hex2byte(digest[2*i], digest[2*i + 1]);

    unsigned char* d_target;
    cudaMalloc(&d_target, 16);
    cudaMemcpy(d_target, h_target, 16, cudaMemcpyHostToDevice);

    /* constants */
    cudaMemcpyToSymbol(d_K, h_K, sizeof(h_K));
    cudaMemcpyToSymbol(d_pw_len, &g_pw_len, sizeof(int));

    /* search-space size */
    uint64_t total_pw = pow64(HOST_CHARSET_SIZE, g_pw_len);

    /* clear early-exit flags */
    g_found = 0; g_idx = 0;

    /* launch */
    const int threads = 256;
    const int blocks  = 1024;

    auto t0 = std::chrono::steady_clock::now();
    brute_kernel<<<blocks, threads>>>(d_target, total_pw);
    cudaDeviceSynchronize();
    double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    /* fetch results */
    int      h_found;
    uint64_t h_idx;
    cudaMemcpyFromSymbol(&h_found, g_found, sizeof(int));
    cudaMemcpyFromSymbol(&h_idx,   g_idx,   sizeof(uint64_t));

    if (h_found) {
        char pw[MAX_PW_LEN + 1];
        idx_to_pw(h_idx, g_pw_len, pw);
        std::cout << "[GPU] " << pw
                  << "  (" << sec << " s, "
                  << (total_pw / sec / 1e9) << " Ghash/s)\n";
    } else {
        std::cout << "[GPU] NOT FOUND ("
                  << sec << " s, "
                  << (total_pw / sec / 1e9) << " Ghash/s)\n";
    }

    cudaFree(d_target);
    return h_found ? 0 : 2;
}
