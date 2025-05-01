#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <sstream>
#include <chrono>

using namespace std::chrono;

void md5_raw(const unsigned char* msg, size_t len, unsigned char d[16]) {
    uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d0 = 0x10325476;
    unsigned char blk[64] = {0};
    std::memcpy(blk, msg, len);
    blk[len] = 0x80;
    uint64_t bits = len * 8ULL;
    std::memcpy(blk + 56, &bits, 8);
    const uint32_t* X = reinterpret_cast<const uint32_t*>(blk);
    const uint32_t K[4] = {0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee};
    for (int i = 0; i < 4; ++i) {
        uint32_t F = (b & c) | (~b & d0);
        F += a + K[i] + X[i];
        a = d0; d0 = c; c = b; b += (F << 7) | (F >> 25);
    }
    a += 0x67452301; b += 0xefcdab89; c += 0x98badcfe; d0 += 0x10325476;
    std::memcpy(d, &a, 4); std::memcpy(d + 4, &b, 4);
    std::memcpy(d + 8, &c, 4); std::memcpy(d + 12, &d0, 4);
}

std::string md5_hash(const std::string& input) {
    unsigned char digest[16];
    md5_raw(reinterpret_cast<const unsigned char*>(input.c_str()), input.length(), digest);
    char hex[33];
    for (int i = 0; i < 16; ++i) std::sprintf(&hex[i * 2], "%02x", digest[i]);
    hex[32] = 0;
    return std::string(hex);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <password_length> <num_threads>\n";
        return 1;
    }

    int length = std::stoi(argv[1]);
    int num_threads = std::stoi(argv[2]);

    if (length < 1 || length > 10) {
        std::cout << "Please use password length between 1 and 10.\n";
        return 1;
    }

    std::string passCode;
    // std::cout << "Enter " << length << "-character password to crack: ";
    std::cin >> passCode;

    if (passCode.length() != length) {
        std::cout << "Error: Password must be exactly " << length << " characters.\n";
        return 1;
    }

    std::string target = md5_hash(passCode);
    const std::string charset = "abcdefghijklmnopqrstuvwxyz";
    int charsetSize = charset.size();
    uint64_t totalCombos = pow(charsetSize, length);

    std::atomic_bool found(false);
    std::string result;

    omp_set_num_threads(num_threads);

    high_resolution_clock::time_point start1 = high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        uint64_t chunkSize = totalCombos / nthreads;
        uint64_t start = tid * chunkSize;
        uint64_t end = (tid == nthreads - 1) ? totalCombos : start + chunkSize;

        // #pragma omp critical
        // {
        //     std::cout << "Thread " << tid << ":\n";
        //     std::cout << "  Chunk size = " << chunkSize << "\n";
        //     std::cout << "  Start      = " << start << "\n";
        //     std::cout << "  End        = " << end << "\n\n";
        // }
        
        std::string attempt(length, 'a');

        for (uint64_t i = start; i < end && !found.load(); ++i) {
            uint64_t temp = i;
            for (int pos = length - 1; pos >= 0; --pos) {
                attempt[pos] = charset[temp % charsetSize];
                temp /= charsetSize;
            }

            if (md5_hash(attempt) == target) {
                if (!found.exchange(true)) {
                    result = attempt;
                }
                break;
            }
        }
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end - start1);

    // Output CSV line
    std::cout << num_threads << "," << time_span.count() << std::endl;

    return 0;
}
