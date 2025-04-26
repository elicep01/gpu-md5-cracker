#ifndef MD5_CPU_H
#define MD5_CPU_H

#include <string>
#include <openssl/md5.h>

static const char CHARSET[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static const int CHARSET_SIZE = sizeof(CHARSET) - 1;
static const int PASSWORD_LEN = 5;  // Fixed to 5 characters

// Convert raw MD5 digest to lowercase hex string
std::string to_hex(const unsigned char* digest);
// Compute MD5 hash of a null-terminated C string
std::string md5_hash(const char* input);
// Brute-force function: returns elapsed seconds, sets found and result
double brute_force_md5(const std::string& target_hash, bool& found, std::string& result);

#endif  // MD5_CPU_H