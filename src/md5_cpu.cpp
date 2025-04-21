#include "md5_cpu.h"
#include <openssl/md5.h>
#include <omp.h>
#include <iostream>
#include <cstring>


std::string to_hex(const unsigned char* digest) {
    char buf[33];
    for (int i = 0; i < 16; ++i)
        std::sprintf(&buf[i * 2], "%02x", (unsigned int)digest[i]);
    return std::string(buf);
}

std::string md5_hash(const char* input) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5((unsigned char*)input, std::strlen(input), digest);
    return to_hex(digest);
}

std::string brute_force_md5(const std::string& target_hash, bool& found, double& elapsed_seconds) {
    char guess[PASSWORD_LEN + 1];
    guess[PASSWORD_LEN] = '\0';
    std::string result;
    found = false;

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) private(guess) shared(found, result)
    for (int i1 = 0; i1 < CHARSET_SIZE; ++i1) {
        for (int i2 = 0; i2 < CHARSET_SIZE; ++i2) {
            if (found) continue;

            for (int i3 = 0; i3 < CHARSET_SIZE; ++i3) {
                for (int i4 = 0; i4 < CHARSET_SIZE; ++i4) {
                    for (int i5 = 0; i5 < CHARSET_SIZE; ++i5) {
                        if (found) continue;

                        guess[0] = CHARSET[i1];
                        guess[1] = CHARSET[i2];
                        guess[2] = CHARSET[i3];
                        guess[3] = CHARSET[i4];
                        guess[4] = CHARSET[i5];

                        std::string hash = md5_hash(guess);
                        if (hash == target_hash) {
                            #pragma omp critical
                            {
                                found = true;
                                result = std::string(guess);
                            }
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    elapsed_seconds = end - start;

    return result;
}


int main() {
    std::string hash;
    std::cout << "Enter MD5 hash of a 5-char alphanumeric password: ";
    std::cin >> hash;

    bool found;
    double elapsed;
    std::string password = brute_force_md5(hash, found, elapsed);

    if (found)
        std::cout << "Password found: " << password << "\n";
    else
        std::cout << "Password not found.\n";

    std::cout << "Time taken: " << elapsed << " seconds\n";
    return 0;
}
