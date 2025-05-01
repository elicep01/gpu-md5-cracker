//****************************************************************
// Optimized GPU MD5 brute-forcer – annotated version (dynamic PASSWORD_LEN)
//****************************************************************
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>

// 1) Configurable charset and length as macros
#define HOST_CHARSET \
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define HOST_CHARSET_SIZE (sizeof(HOST_CHARSET)-1)
#define PASSWORD_LEN      6

/* ---------- early-exit globals (managed) ---------- */
__device__ __managed__ volatile int      g_found = 0;
__device__ __managed__ uint64_t          g_idx   = 0;

/* ---------- constant memory ---------- */
__device__ __constant__ uint32_t d_K[64];
__device__ __constant__ int d_r[64] = {
   7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
   5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
   4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
   6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
};
__device__ __constant__ char d_CHARSET[HOST_CHARSET_SIZE+1] = HOST_CHARSET;

/* ---------- host-to-device constant upload ---------- */
void upload_constants() {
    static const uint32_t h_K[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
    };
    cudaMemcpyToSymbol(d_K, h_K, sizeof(h_K));
}

/* ---------- utility functions ---------- */
__device__ __forceinline__ uint32_t leftrotate(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

__host__ unsigned char hex2byte(char hi, char lo) {
    auto val = [&](char c)->int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    return (val(hi) << 4) | val(lo);
}

/* ---------- single-block MD5 (dynamic-length version) ---------- */
__device__ __forceinline__ void md5_single(const char* in, unsigned char dig[16]) {
    uint32_t M[16];
    // zero-out block
    #pragma unroll
    for (int i = 0; i < 16; ++i) M[i] = 0;

    // copy PASSWORD_LEN bytes
    #pragma unroll
    for (int i = 0; i < PASSWORD_LEN; ++i) {
        int word = i >> 2;
        int shift = (i & 3) * 8;
        M[word] |= ((uint32_t)in[i] << shift);
    }
    // append 0x80 bit
    {
        int i = PASSWORD_LEN;
        int word = i >> 2;
        int shift = (i & 3) * 8;
        M[word] |= (0x80u << shift);
    }
    // message length in bits
    M[14] = (uint32_t)PASSWORD_LEN * 8;
    M[15] = 0;

    uint32_t a = 0x67452301, b = 0xefcdab89,
             c = 0x98badcfe, d = 0x10325476;

    #pragma unroll 64
    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if (i < 16)       { F = (b & c) | (~b & d);       g = i;      }
        else if (i < 32)  { F = (d & b) | (~d & c);       g = (5*i+1)&15; }
        else if (i < 48)  { F = b ^ c ^ d;                g = (3*i+5)&15; }
        else              { F = c ^ (b | ~d);            g = (7*i)&15;   }
        F += a + d_K[i] + M[g];
        a = d; d = c; c = b; b += leftrotate(F, d_r[i]);
    }
    a += 0x67452301; b += 0xefcdab89;
    c += 0x98badcfe; d += 0x10325476;

    uint32_t regs[4] = {a,b,c,d};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dig[4*i]   = regs[i] & 0xFF;
        dig[4*i+1] = (regs[i] >> 8) & 0xFF;
        dig[4*i+2] = (regs[i] >> 16) & 0xFF;
        dig[4*i+3] = (regs[i] >> 24) & 0xFF;
    }
}

/* ---------- brute-force kernel ---------- */
__global__ __launch_bounds__(256,4)
void brute7(const unsigned char* target_bin, uint64_t total) {
    if (g_found) return;
    uint64_t stride = blockDim.x * (uint64_t)gridDim.x;
    uint64_t idx    = blockIdx.x * blockDim.x + threadIdx.x;

    uint4 t4 = *((uint4*)target_bin);
    unsigned char dig[16];
    char pw[PASSWORD_LEN+1]; pw[PASSWORD_LEN] = '\0';

    for (; idx < total; idx += stride) {
        if (g_found) return;
        uint64_t v = idx;
        #pragma unroll
        for (int p = PASSWORD_LEN-1; p >= 0; --p) {
            pw[p] = d_CHARSET[v % HOST_CHARSET_SIZE];
            v /= HOST_CHARSET_SIZE;
        }
        md5_single(pw, dig);
        uint4 d4 = *((uint4*)dig);
        if (d4.x==t4.x && d4.y==t4.y && d4.z==t4.z && d4.w==t4.w) {
            if (atomicCAS((int*)&g_found,0,1)==0) g_idx = idx;
            return;
        }
    }
}

/* ---------- host driver ---------- */
int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <32-char MD5 hex>\n", argv[0]);
        return 1;
    }
    upload_constants();

    unsigned char h_target[16];
    for (int i = 0; i < 16; ++i) {
        h_target[i] = hex2byte(argv[1][2*i], argv[1][2*i+1]);
    }
    unsigned char* d_target;
    cudaMalloc(&d_target, 16);
    cudaMemcpy(d_target, h_target, 16, cudaMemcpyHostToDevice);

    uint64_t total = 1;
    for (int i = 0; i < PASSWORD_LEN; ++i) total *= HOST_CHARSET_SIZE;

    const int blocks = 1024, threads = 256;
    auto t0 = std::chrono::steady_clock::now();
    brute7<<<blocks,threads>>>(d_target, total);
    cudaDeviceSynchronize();
    double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    int h_found = 0;
    uint64_t h_idx = 0;
    cudaMemcpyFromSymbol(&h_found, g_found, sizeof(int));
    cudaMemcpyFromSymbol(&h_idx,   g_idx,   sizeof(uint64_t));

    if (h_found) {
        char pw[PASSWORD_LEN+1];
        uint64_t v = h_idx;
        for (int p = PASSWORD_LEN-1; p >= 0; --p) {
            pw[p] = HOST_CHARSET[v % HOST_CHARSET_SIZE];
            v /= HOST_CHARSET_SIZE;
        }
        pw[PASSWORD_LEN] = '\0';
        printf("Password found : %s\n", pw);
    } else {
        puts("Password NOT found.");
    }
    printf("GPU elapsed     : %.6f s  (%.2f Ghash/s)\n", sec, total/sec/1e9);
    cudaFree(d_target);
    return 0;
}
