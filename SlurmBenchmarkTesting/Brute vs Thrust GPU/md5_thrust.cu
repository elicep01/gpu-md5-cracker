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
    for (int i = 0; i < 16; ++i)
        sscanf(argv[1] + 2 * i, "%2hhx", &h_target[i]);
    uint4 target4;
    memcpy(&target4, h_target, 16);

    // 3) compute total password space
    uint64_t total_pw = 1;
    for (int i = 0; i < PASSWORD_LEN; ++i)
        total_pw *= HOST_CHARSET_SIZE;

    // 4) launch Thrust search in chunks
    const uint64_t CHUNK_SIZE = 1ULL << 30; // ~1 billion
    CrackFunctor functor{target4, total_pw};

    printf("=== Running thrust_crack GPU cracker ===\n");
    fflush(stdout);

    g_found = 0;
    g_idx = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (uint64_t i = 0; i < total_pw; i += CHUNK_SIZE) {
        uint64_t current_size = std::min(CHUNK_SIZE, total_pw - i);
        thrust::for_each_n(thrust::device,
            thrust::make_counting_iterator<uint64_t>(i),
            current_size,
            functor);
        if (g_found) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_s = elapsed_ms / 1000.0;

    // 5) copy back result
    if (g_found) {
        uint64_t idx = g_idx;
        char pw[PASSWORD_LEN + 1];
        pw[PASSWORD_LEN] = '\0';
        for (int p = PASSWORD_LEN - 1; p >= 0; --p) {
            pw[p] = h_CHARSET[idx % HOST_CHARSET_SIZE];
            idx /= HOST_CHARSET_SIZE;
        }
        double ghashes = (double)g_idx / 1e9 / elapsed_s;
        printf("Password found : %s\n", pw);
        printf("GPU elapsed     : %.6f s  (%.2f Ghash/s)\n", elapsed_s, ghashes);
    } else {
        printf("Password NOT found\n");
        printf("GPU elapsed     : %.6f s\n", elapsed_s);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
