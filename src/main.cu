/******************************************************************
*  GPU MD5 brute-forcer – 7-char alphanumeric (ECE 759 project)   *
*  Tasks 1 + 2 + 3:                                               *
*    • device-wide early-exit (volatile flag + atomicCAS)         *
*    • no per-thread printf – host prints once                    *
*    • basic timing & throughput                                  *
******************************************************************/
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>

/* ---------- global early-exit flag (managed) ---------- */
__device__ __managed__ volatile int      g_found = 0;   // 0 = not yet, 1 = found
__device__ __managed__ uint64_t          g_idx   = 0;   // winning index (base-62)

/* ---------- search-space parameters ---------- */
#define HOST_CHARSET_SIZE 62
#define PASSWORD_LEN      7

/* ---------- constant memory ---------- */
__device__ __constant__ char d_CHARSET[HOST_CHARSET_SIZE + 1] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

__device__ __constant__ unsigned int K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

/* ---------- helpers ---------- */
__device__ unsigned int leftrotate(unsigned int x, unsigned int c)
{ return (x << c) | (x >> (32 - c)); }

__device__ bool str_eq(const char* a, const char* b) {
    int i = 0;  while (a[i] && b[i]) { if (a[i] != b[i]) return false; ++i; }
    return a[i] == b[i];
}

/* host helper: convert base-62 index → 7-char string */
__host__ void idx_to_pw(uint64_t idx, char* pw)
{
    static const char* cs =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int p = PASSWORD_LEN - 1; p >= 0; --p) {
        pw[p] = cs[idx % HOST_CHARSET_SIZE];
        idx  /= HOST_CHARSET_SIZE;
    }
    pw[PASSWORD_LEN] = '\0';
}

/* ---------- single-block MD5 (≤55 bytes) ---------- */
__device__ void md5_single(const char* input, unsigned char digest[16])
{
    unsigned char msg[64] = {0};
    int len = 0; while (input[len] && len < 64) ++len;
    for (int i = 0; i < len; ++i) msg[i] = input[i];
    msg[len] = 0x80;
    uint64_t bit_len = static_cast<uint64_t>(len) * 8ULL;
    for (int i = 0; i < 8; ++i) msg[56+i] = (bit_len >> (8*i)) & 0xFF;

    unsigned int a = 0x67452301, b = 0xefcdab89,
                 c = 0x98badcfe, d = 0x10325476;

    unsigned int M[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        M[i] =  msg[4*i] |
               (msg[4*i+1] << 8) |
               (msg[4*i+2] <<16) |
               (msg[4*i+3] <<24);

    const int r[64] = {7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
                       5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
                       4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
                       6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        unsigned int F, g;
        if (i < 16)        { F = (b & c) | (~b & d); g = i; }
        else if (i < 32)   { F = (d & b) | (~d & c); g = (5*i + 1) & 15; }
        else if (i < 48)   { F = b ^ c ^ d;          g = (3*i + 5) & 15; }
        else               { F = c ^ (b | ~d);       g = (7*i)     & 15; }
        unsigned int tmp = d;
        d = c;  c = b;
        unsigned int sum = a + F + K[i] + M[g];
        b += leftrotate(sum, r[i]);
        a = tmp;
    }
    unsigned int regs[4] = {a + 0x67452301, b + 0xefcdab89,
                            c + 0x98badcfe, d + 0x10325476};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        digest[4*i]   =  regs[i]        & 0xFF;
        digest[4*i+1] = (regs[i] >> 8)  & 0xFF;
        digest[4*i+2] = (regs[i] >> 16) & 0xFF;
        digest[4*i+3] = (regs[i] >> 24) & 0xFF;
    }
}

__device__ void to_hex_device(const unsigned char* digest, char* out)
{
    const char* hex = "0123456789abcdef";
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[2*i]   = hex[(digest[i] >> 4) & 0xF];
        out[2*i+1] = hex[ digest[i]       & 0xF];
    }
    out[32] = '\0';
}

/* ---------- brute-force kernel ---------- */
__global__ void brute7(const char* target)
{
    if (g_found) return;                       // late warps bail

    uint64_t total = 1;
    for (int i = 0; i < PASSWORD_LEN; ++i) total *= HOST_CHARSET_SIZE;

    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t idx    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    char  guess[PASSWORD_LEN + 1];  guess[PASSWORD_LEN] = '\0';
    unsigned char digest[16];  char hex[33];

    for (; idx < total; idx += stride) {
        if (g_found) return;

        uint64_t v = idx;
        #pragma unroll
        for (int p = PASSWORD_LEN - 1; p >= 0; --p) {
            guess[p] = d_CHARSET[v % HOST_CHARSET_SIZE];
            v /= HOST_CHARSET_SIZE;
        }

        md5_single(guess, digest);
        to_hex_device(digest, hex);

        if (str_eq(hex, target)) {                 // HIT
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                g_idx = idx;                       // record first winner
                __threadfence();                   // flush to global mem
            }
            return;                                // thread exits
        }
    }
}

/* ---------- host driver ---------- */
int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s <32-char MD5 hex>\n", argv[0]);
        return 1;
    }

    char* d_target;
    cudaMalloc(&d_target, 33);
    cudaMemcpy(d_target, argv[1], 33, cudaMemcpyHostToDevice);

    const uint64_t total_pw = 1'028'071'702'528ULL;     // 62^7
    const int threads = 256, blocks = 1024;

    auto t0 = std::chrono::steady_clock::now();
    brute7<<<blocks, threads>>>(d_target);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    int      h_found = 0;
    uint64_t h_idx   = 0;
    cudaMemcpyFromSymbol(&h_found, g_found, sizeof(int));
    cudaMemcpyFromSymbol(&h_idx,   g_idx,   sizeof(uint64_t));

    if (h_found) {
        char pw[PASSWORD_LEN + 1];
        idx_to_pw(h_idx, pw);
        printf("Password found : %s\n", pw);
    } else {
        puts("Password NOT found.");
    }
    printf("GPU elapsed     : %.3f s  (%.2f Ghash/s)\n",
           sec, total_pw / sec / 1e9);

    cudaFree(d_target);
    return 0;
}
