#pragma once
#include <openssl/sha.h>
static inline int mbedtls_sha256(const unsigned char *data,size_t n,unsigned char *out,int is224) { (void)is224;return SHA256(data,n,out)?0:-1; }
