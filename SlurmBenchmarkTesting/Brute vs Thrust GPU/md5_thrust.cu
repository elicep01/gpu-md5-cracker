#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#define HOST_CHARSET_SIZE 62
#define PASSWORD_LEN      7

// unified managed flags (as in your kernel)
__device__ __managed__ volatile int      g_found = 0;
__device__ __managed__ uint64_t          g_idx   = 0;

// constant memory copies of charset and K[]
__device__ __constant__ char     d_CHARSET[HOST_CHARSET_SIZE+1];
__device__ __constant__ uint32_t d_K[64];

// host definitions of charset and K[]
static const char   h_CHARSET[HOST_CHARSET_SIZE+1]
    = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static const uint32_t h_K[64] = {
    /* … same 64 MD5 constants … */
    0xd76aa478,0xe8c7b756, /* … etc … */ 0x2ad7d2bb,0xeb86d391
};

// helper to rotate left
__device__ __forceinline__ uint32_t leftrotate(uint32_t x,int c){
    return (x<<c)|(x>>(32-c));
}

// device‐inline single‐block MD5 (exactly your md5_single)
__device__ void md5_single(const char* in, unsigned char dig[16]) {
    uint32_t M[16];
    // pack input + padding into M[0..1], zeros M[2..13], len in M[14]
    M[0] =  (uint32_t)in[0] | ((uint32_t)in[1]<<8)
          | ((uint32_t)in[2]<<16) | ((uint32_t)in[3]<<24);
    M[1] =  (uint32_t)in[4] | ((uint32_t)in[5]<<8)
          | ((uint32_t)in[6]<<16) | (0x80u<<24);
    #pragma unroll
    for (int i=2;i<14;++i) M[i]=0u;
    M[14] = PASSWORD_LEN*8; M[15]=0u;

    uint32_t a=0x67452301, b=0xefcdab89,
             c=0x98badcfe, d=0x10325476;
    const int r[64] = {
       7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
       5, 9,14,20,5, 9,14,20,5, 9,14,20,5, 9,14,20,
       4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
       6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
    };

    #pragma unroll
    for (int i=0;i<64;++i) {
        uint32_t F,g;
        if (i<16)      { F=(b&c)|(~b&d);   g=i; }
        else if (i<32) { F=(d&b)|(~d&c);   g=(5*i+1)&15; }
        else if (i<48) { F=b^c^d;          g=(3*i+5)&15; }
        else           { F=c^(b|~d);       g=(7*i)&15; }
        uint32_t tmp=d; d=c; c=b;
        F += a + d_K[i] + M[g];
        b += leftrotate(F,r[i]);
        a = tmp;
    }
    // add to initial state
    a+=0x67452301; b+=0xefcdab89;
    c+=0x98badcfe; d+=0x10325476;

    uint32_t regs[4]={a,b,c,d};
    #pragma unroll
    for(int i=0;i<4;++i){
        dig[4*i  ] =  regs[i]        & 0xFF;
        dig[4*i+1] = (regs[i] >>  8) & 0xFF;
        dig[4*i+2] = (regs[i] >> 16) & 0xFF;
        dig[4*i+3] = (regs[i] >> 24) & 0xFF;
    }
}

// functor run by thrust::for_each_n
struct CrackFunctor {
    const uint4 target4;      // target digest as uint4
    const uint64_t total_pw;

    __device__
    void operator()(uint64_t idx) const {
        if (g_found) return;

        // strided loop not needed—thrust hands each idx once
        char pw[PASSWORD_LEN+1];
        pw[PASSWORD_LEN]='\0';
        uint64_t v = idx;
        #pragma unroll
        for (int p=PASSWORD_LEN-1; p>=0; --p) {
            pw[p] = d_CHARSET[v % HOST_CHARSET_SIZE];
            v /= HOST_CHARSET_SIZE;
        }

        unsigned char dig[16];
        md5_single(pw, dig);

        // compare in 128‐bit chunks
        uint4* d4 = (uint4*)dig;
        if (d4->x==target4.x && d4->y==target4.y
         && d4->z==target4.z && d4->w==target4.w) {
            if (atomicCAS((int*)&g_found,0,1)==0)
                g_idx = idx;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <32-hex MD5>\n", argv[0]);
        return 1;
    }

    // 1) copy constants into device constant memory
    cudaMemcpyToSymbol(d_CHARSET, h_CHARSET, sizeof(h_CHARSET));
    cudaMemcpyToSymbol(d_K,       h_K,       sizeof(h_K));

    // 2) parse target hex into uint4
    unsigned char h_target[16];
    for (int i=0;i<16;++i)
        sscanf(argv[1]+2*i, "%2hhx", &h_target[i]);
    uint4 target4;
    memcpy(&target4, h_target, 16);

    // 3) compute total password space
    uint64_t total_pw = 1;
    for (int i=0;i<PASSWORD_LEN;++i)
        total_pw *= HOST_CHARSET_SIZE;

    // 4) launch Thrust search
    thrust::counting_iterator<uint64_t> start(0);
    CrackFunctor functor{target4, total_pw};

    // this will schedule enough threads to cover [0,total_pw)
    thrust::for_each_n(thrust::device, start, total_pw, functor);

    // 5) copy back result
    if (g_found) {
        // reconstruct on host
        uint64_t idx = g_idx;
        char pw[PASSWORD_LEN+1];
        for(int p=PASSWORD_LEN-1;p>=0;--p){
            pw[p] = h_CHARSET[idx % HOST_CHARSET_SIZE];
            idx /= HOST_CHARSET_SIZE;
        }
        printf("Password found: %s  (idx=%llu)\n",
               pw, (unsigned long long)g_idx);
    } else {
        printf("Password NOT found\n");
    }

    return 0;
}
