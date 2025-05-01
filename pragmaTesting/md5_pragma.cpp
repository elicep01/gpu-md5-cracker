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
    char *password = argv[1];
    int passwordLength = strlen(password);
    int threadCount = omp_get_max_threads();

    std::string hashedPassword = md5_hash(password);
    const std::string charSet = "abcdefghijklmnopqrstuvwxyz";
    int charSetSize = charSet.size();
    uint64_t possiblePasswords = pow(charSetSize, passwordLength);

    std::atomic_bool crackedPassword(false);

    omp_set_num_threads(threadCount);

    high_resolution_clock::time_point start = high_resolution_clock::now();

    #pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        int threadCount = omp_get_num_threads();

        uint64_t passwordsToTry = possiblePasswords / threadCount;
        uint64_t startPassword = threadNum * passwordsToTry;
        uint64_t endPassword = startPassword + passwordsToTry;
        if (threadNum == threadCount - 1) {
            endPassword = possiblePasswords;
        }
        
        std::string crackAttempt(passwordLength, 'a');

        for (uint64_t i = startPassword; i < endPassword && !crackedPassword.load(); ++i) {
            uint64_t currentPassword = i;
            for (int pos = passwordLength - 1; pos >= 0; --pos) {
                crackAttempt[pos] = charSet[currentPassword % charSetSize];
                currentPassword /= charSetSize;
            }

            if (md5_hash(crackAttempt) == hashedPassword) {
                crackedPassword.exchange(true);
            }
        }
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> time = duration_cast<duration<double>>(end - start);

    std::cout << time.count() << std::endl;

    return 0;
}
