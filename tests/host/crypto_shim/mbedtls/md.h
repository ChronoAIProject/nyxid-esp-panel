#pragma once
// Host-only adapter: exercise firmware signing logic against OpenSSL primitives.
#include <openssl/hmac.h>
#define MBEDTLS_MD_SHA256 1
typedef EVP_MD mbedtls_md_info_t;
typedef struct { HMAC_CTX *ctx; } mbedtls_md_context_t;
static inline void mbedtls_md_init(mbedtls_md_context_t *c) { c->ctx=NULL; }
static inline const mbedtls_md_info_t *mbedtls_md_info_from_type(int type) { (void)type; return EVP_sha256(); }
static inline int mbedtls_md_setup(mbedtls_md_context_t *c, const mbedtls_md_info_t *m, int h) { (void)m;(void)h; c->ctx=HMAC_CTX_new(); return c->ctx?0:-1; }
static inline int mbedtls_md_hmac_starts(mbedtls_md_context_t *c,const unsigned char *k,size_t n) { return HMAC_Init_ex(c->ctx,k,(int)n,EVP_sha256(),NULL)?0:-1; }
static inline int mbedtls_md_hmac_update(mbedtls_md_context_t *c,const unsigned char *b,size_t n) { return HMAC_Update(c->ctx,b,n)?0:-1; }
static inline int mbedtls_md_hmac_finish(mbedtls_md_context_t *c,unsigned char *b) { unsigned n=0;return HMAC_Final(c->ctx,b,&n)?0:-1; }
static inline void mbedtls_md_free(mbedtls_md_context_t *c) { HMAC_CTX_free(c->ctx); }
