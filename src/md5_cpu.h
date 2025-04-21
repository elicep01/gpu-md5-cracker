#ifndef MD5_CPU_H
#define MD5_CPU_H

#include <string>

const char CHARSET[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const int CHARSET_SIZE = sizeof(CHARSET) - 1;
const int PASSWORD_LEN = 5; // Fixed to 5 characters

// Hashing helpers
std::string to_hex(const unsigned char* digest);
std::string md5_hash(const char* input);

// Brute-force function
std::string brute_force_md5(const std::string& target_hash, bool& found, double& elapsed_seconds);

#endif
