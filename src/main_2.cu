__device__ void generate_password(uint64_t idx, char* pw, const char* charset) {
    for (int char_pos = PASSWORD_LEN - 1; char_pos >= 0; --char_pos) {
        pw[char_pos] = charset[idx % HOST_CHARSET_SIZE];
        idx /= HOST_CHARSET_SIZE;
    }
    pw[PASSWORD_LEN] = '\0';
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void brute7(const unsigned char* target_bin) {
    __shared__ char s_CHARSET[HOST_CHARSET_SIZE];
    if (threadIdx.x < HOST_CHARSET_SIZE) {
        s_CHARSET[threadIdx.x] = d_CHARSET[threadIdx.x];
    }
    __syncthreads();

    if (g_found) return;

    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    char pw[PASSWORD_LEN + 1];
    unsigned char dig[16];
    bool found = false;

    for (; idx < total; idx += stride) {
        if (found || g_found) return;

        generate_password(idx, pw, s_CHARSET);
        md5_single(pw, dig);

        uint4* d4 = (uint4*)dig;
        uint4* t4 = (uint4*)target_bin;
        if (d4->x == t4->x && d4->y == t4->y &&
            d4->z == t4->z && d4->w == t4->w) {
            found = true;
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                g_idx = idx;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <32-char MD5 hex>\n", argv[0]);
        return 1;
    }

    unsigned char h_target[16];
    for (int i = 0; i < 16; ++i) {
        h_target[i] = hex2byte(argv[1][2 * i], argv[1][2 * i + 1]);
    }

    unsigned char* d_target;
    cudaMalloc(&d_target, 16);
    cudaMemcpy(d_target, h_target, 16, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_K, h_K, sizeof(h_K));

    const uint64_t total_pw = pow64(HOST_CHARSET_SIZE, PASSWORD_LEN);
    const int blocks = BLOCKS_PER_GRID, threads = THREADS_PER_BLOCK;

    auto t0 = std::chrono::steady_clock::now();
    brute7<<<blocks, threads>>>(d_target);
    cudaDeviceSynchronize();
    double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    int h_found = 0;
    uint64_t h_idx = 0;
    cudaMemcpyFromSymbol(&h_found, g_found, sizeof(int));
    cudaMemcpyFromSymbol(&h_idx, g_idx, sizeof(uint64_t));

    if (h_found) {
        char pw[PASSWORD_LEN + 1];
        idx_to_pw(h_idx, pw);
        printf("Password found: %s\n", pw);
    } else {
        printf("Password NOT found.\n");
    }

    printf("Elapsed time: %.6f seconds\n", sec);
    printf("Performance: %.2f Ghash/s\n", total_pw / sec / 1e9);

    cudaFree(d_target);
    return 0;
}