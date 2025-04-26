#include "md5_cpu.h"
#include <omp.h>
#include <iostream>
#include <cstring>
#include <algorithm>  
#include <cctype>     

// Convert raw MD5 digest to lowercase hex string
std::string to_hex(const unsigned char* digest) {
    char buf[2 * MD5_DIGEST_LENGTH + 1];
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        std::sprintf(&buf[2 * i], "%02x", static_cast<unsigned int>(digest[i]));
    }
    buf[2 * MD5_DIGEST_LENGTH] = '\0';
    return std::string(buf);
}

// Compute MD5 hash of a null-terminated C string
std::string md5_hash(const char* input) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(input), std::strlen(input), digest);
    return to_hex(digest);
}

// Brute-force a 5-character alphanumeric password
// Returns elapsed time; sets 'found' and 'result' when match is found
double brute_force_md5(const std::string& target_hash, bool& found, std::string& result) {
    char guess[PASSWORD_LEN + 1] = {0};
    found = false;
    double start = omp_get_wtime();

    #pragma omp parallel for private(guess) shared(found, result)
    for (int i1 = 0; i1 < CHARSET_SIZE; ++i1) {
        for (int i2 = 0; i2 < CHARSET_SIZE && !found; ++i2) {
            for (int i3 = 0; i3 < CHARSET_SIZE && !found; ++i3) {
                for (int i4 = 0; i4 < CHARSET_SIZE && !found; ++i4) {
                    for (int i5 = 0; i5 < CHARSET_SIZE && !found; ++i5) {
                        guess[0] = CHARSET[i1];
                        guess[1] = CHARSET[i2];
                        guess[2] = CHARSET[i3];
                        guess[3] = CHARSET[i4];
                        guess[4] = CHARSET[i5];
                        std::string h = md5_hash(guess);
                        if (h == target_hash) {
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
    return end - start;
}

int main() {
    std::string hash_input;
    std::cout << "Enter MD5 hash of a 5-char alphanumeric password: ";
    std::cin >> hash_input;

    // Normalize input to lowercase
    std::transform(hash_input.begin(), hash_input.end(), hash_input.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Quick check for "aaaaa" (first in search space)
    // Sanity - check remove this later
    if (hash_input == md5_hash("aaaaa")) {
        std::cout << "Password found: aaaaa\n";
        std::cout << "Time taken: 0.00 seconds\n";
        return 0;
    }

    bool found;
    std::string password;
    double elapsed = brute_force_md5(hash_input, found, password);

    if (found) {
        std::cout << "Password found: " << password << "\n";
    } else {
        std::cout << "Password not found.\n";
    }
    std::cout << "Time taken: " << elapsed << " seconds\n";
    return 0;
}
