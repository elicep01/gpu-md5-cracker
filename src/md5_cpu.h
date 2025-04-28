#ifndef MD5_CPU_H
#define MD5_CPU_H

#include <string>
#include <cstdint>

/* --- minimal MD5 prototype (public-domain) */
void md5_raw(const unsigned char* data, size_t len, unsigned char digest[16]);

/* --- brute-force parameters --- */
static const char CHARSET[] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static const int  CHARSET_SIZE = sizeof(CHARSET) - 1;
static const int  PASSWORD_LEN = 7;

/* --- helpers --- */
std::string to_hex(const unsigned char* digest);
std::string md5_hash(const char* input);

/* brute-force a 7-char password; returns seconds, sets found+result */
double brute_force_md5(const std::string& target_hash,
                       bool& found, std::string& result,
                       int num_threads);

#endif // MD5_CPU_H
