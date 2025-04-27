#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Host charset size and password length
#define HOST_CHARSET_SIZE 62
#define PASSWORD_LEN 7

// Device charset and constants in constant memory
__device__ __constant__ char d_CHARSET[HOST_CHARSET_SIZE+1] =
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

// Rotate left
__device__ unsigned int leftrotate(unsigned int x, unsigned int c) {
    return (x << c) | (x >> (32 - c));
}

// Compare two device strings
__device__ bool str_eq(const char* a, const char* b) {
    int i = 0;
    while (a[i] && b[i]) {
        if (a[i] != b[i]) return false;
        ++i;
    }
    return a[i] == b[i];
}

// MD5 for single block
__device__ void md5_single(const char* input, unsigned char digest[16]) {
    unsigned char msg[64] = {0};
    int len = 0; while(input[len] && len < 64) ++len;
    for(int i=0;i<len;++i) msg[i] = input[i];
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8ULL;
    for(int i=0;i<8;++i) msg[56+i] = (bit_len >> (8*i)) & 0xFF;

    unsigned int a0=0x67452301, b0=0xefcdab89, c0=0x98badcfe, d0=0x10325476;
    unsigned int M[16];
    for(int i=0;i<16;++i) {
        M[i] = (unsigned int)msg[4*i] |
               ((unsigned int)msg[4*i+1]<<8) |
               ((unsigned int)msg[4*i+2]<<16) |
               ((unsigned int)msg[4*i+3]<<24);
    }
    const int r[64] = {7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
                       5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
                       4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
                       6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};
    unsigned int A=a0,B=b0,C=c0,D=d0;
    for(int i=0;i<64;++i) {
        unsigned int F,g;
        if(i<16){F=(B&C)|((~B)&D);g=i;} else if(i<32){F=(D&B)|((~D)&C);g=(5*i+1)%16;} 
        else if(i<48){F=B^C^D;g=(3*i+5)%16;} else {F=C^(B|(~D));g=(7*i)%16;}
        unsigned int tmp=D; D=C; C=B;
        unsigned int sum=A+F+K[i]+M[g];
        B=B+leftrotate(sum,r[i]); A=tmp;
    }
    a0+=A; b0+=B; c0+=C; d0+=D;
    unsigned int regs[4]={a0,b0,c0,d0};
    for(int i=0;i<4;++i) {
        digest[4*i]   = regs[i] & 0xFF;
        digest[4*i+1] = (regs[i]>>8)&0xFF;
        digest[4*i+2] = (regs[i]>>16)&0xFF;
        digest[4*i+3] = (regs[i]>>24)&0xFF;
    }
}

// Digest to hex
__device__ void to_hex_device(const unsigned char* digest, char* out) {
    const char* hex="0123456789abcdef";
    for(int i=0;i<16;++i){out[2*i]=hex[(digest[i]>>4)&0xF];out[2*i+1]=hex[digest[i]&0xF];}
    out[32]='\0';
}

// Brute-force 7-char passwords with grid-stride loop
__global__ void brute7(const char* target) {
    uint64_t total=1;
    for(int i=0;i<PASSWORD_LEN;++i) total*=HOST_CHARSET_SIZE;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    char guess[PASSWORD_LEN+1]; guess[PASSWORD_LEN]='\0';
    unsigned char digest[16]; char hex[33];
    for(uint64_t idx=tid; idx<total; idx+=stride) {
        uint64_t v=idx;
        for(int pos=PASSWORD_LEN-1; pos>=0; --pos) {
            int c = v % HOST_CHARSET_SIZE;
            guess[pos] = d_CHARSET[c];
            v /= HOST_CHARSET_SIZE;
        }
        md5_single(guess,digest);
        to_hex_device(digest,hex);
        if(str_eq(hex,target)) printf("Found: %s -> %s\n",guess,hex);
    }
}

int main(int argc, char** argv) {
    if(argc!=2){printf("Usage: %s <32-char MD5>\n",argv[0]);return 1;}
    char* d_target; cudaMalloc(&d_target,33);
    cudaMemcpy(d_target,argv[1],33,cudaMemcpyHostToDevice);
    int threads=256, blocks=1024;
    brute7<<<blocks,threads>>>(d_target);
    cudaDeviceSynchronize(); cudaFree(d_target);
    return 0;
}
