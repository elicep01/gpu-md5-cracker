#ifndef MD5_CPU_H
#define MD5_CPU_H
// Prevent multiple inclusion of this header file

#include <string>   // std::string for returning text-based results
#include <cstdint>  // fixed-width integer types (e.g., uint32_t)

/* --- minimal MD5 prototype (public-domain) */
// Computes the raw 16-byte MD5 digest of `data` of length `len`
void md5_raw(const unsigned char* data, size_t len, unsigned char digest[16]);

/* --- brute-force parameters --- */
// Characters used when generating candidate passwords
static const char CHARSET[] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
// Number of characters in CHARSET (exclude terminating '\0')
static const int  CHARSET_SIZE = sizeof(CHARSET) - 1;
// Length of passwords to generate/check
static const int  PASSWORD_LEN = 7;

/* --- helpers --- */
// Convert 16-byte binary digest to a 32-character hexadecimal string
std::string to_hex(const unsigned char* digest);
// Wrapper: compute MD5 of a C-style string and return its hex representation
std::string md5_hash(const char* input);

/* brute-force a 7-char password; returns seconds, sets found+result */
// Tries every combination in CHARSET^PASSWORD_LEN using `num_threads` threads.
// On success, sets `found` to true and writes the password into `result`.
double brute_force_md5(const std::string& target_hash,
                       bool& found, std::string& result,
                       int num_threads);

#endif // MD5_CPU_H  // end include guard