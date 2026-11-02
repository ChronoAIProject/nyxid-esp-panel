#include "nyx_security.h"
#include "mbedtls/md.h"
#include "mbedtls/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

static struct { char nonce[37]; int64_t expires; } seen[128];
static int hex_digit(char c) {
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return c-'a'+10;
    if (c>='A' && c<='F') return c-'A'+10;
    return -1;
}
static bool secret_bytes(const char *hex, uint8_t out[32]) {
    if (!hex || strlen(hex)!=64) return false;
    for (size_t i=0;i<32;++i) {
        int a=hex_digit(hex[i*2]), b=hex_digit(hex[i*2+1]);
        if (a<0 || b<0) return false;
        out[i]=(uint8_t)((a<<4)|b);
    }
    return true;
}
static const char *string(cJSON *root, const char *key) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, key);
    return cJSON_IsString(v) ? v->valuestring : "";
}
static bool freshness(const char *nonce, int64_t stamp) {
    int64_t now = time(NULL);
    if (strlen(nonce) != 36 || now < 1700000000 || stamp < now - 60 || stamp > now + 60) return false;
    for (size_t i = 0; i < 128; ++i)
        if (seen[i].expires >= now && !strcmp(seen[i].nonce, nonce)) return false;
    for (size_t i = 0; i < 128; ++i) {
        if (seen[i].expires < now) {
            snprintf(seen[i].nonce, sizeof(seen[i].nonce), "%s", nonce);
            seen[i].expires = stamp + 60; return true;
        }
    }
    return false;
}
static bool digest_matches(const uint8_t *digest, const char *hex) {
    if (strlen(hex) != 64) return false;
    unsigned diff = 0;
    for (size_t i = 0; i < 32; ++i) {
        char tmp[3] = { hex[2*i], hex[2*i+1], 0 }, *end;
        unsigned long b = strtoul(tmp, &end, 16);
        if (*end) return false;
        diff |= digest[i] ^ b;
    }
    return !diff;
}
static int64_t rfc3339(const char *stamp) {
    int y, m, d, h, minute, second, n = 0;
    if (sscanf(stamp,"%d-%d-%dT%d:%d:%d%n", &y,&m,&d,&h,&minute,&second,&n) != 6 ||
        y < 2020 || m < 1 || m > 12 || d < 1 || d > 31 || h > 23 || minute > 59 || second > 59) return 0;
    const char *suffix = stamp + n;
    if (*suffix == '.') { ++suffix; while (*suffix >= '0' && *suffix <= '9') ++suffix; }
    if (strcmp(suffix,"Z") && strcmp(suffix,"+00:00")) return 0;
    y -= m <= 2;
    int era = y / 400, year = y - era * 400;
    int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    int days = era * 146097 + year * 365 + year / 4 - year / 100 + doy - 719468;
    return (int64_t)days * 86400 + h * 3600 + minute * 60 + second;
}
bool nyx_security_verify_proxy(cJSON *root, const char *secret) {
    if (!secret || !*secret) return true; // Legacy key-only deployments authenticate the WSS session.
    if (cJSON_HasObjectItem(root, "signature_version")) return false; // v2 destinations are not advertised.
    uint8_t key[32];
    if (!secret_bytes(secret,key)) return false;
    const char *fields[] = {string(root,"timestamp"), string(root,"nonce"), string(root,"method"),
        string(root,"path"), string(root,"query"), string(root,"body")};
    mbedtls_md_context_t mac;
    mbedtls_md_init(&mac);
    bool ok = mbedtls_md_setup(&mac,mbedtls_md_info_from_type(MBEDTLS_MD_SHA256),1)==0 &&
        mbedtls_md_hmac_starts(&mac,key,sizeof(key))==0;
    for (size_t i = 0; ok && i < 6; ++i) {
        if (i) ok = mbedtls_md_hmac_update(&mac,(const uint8_t *)"\n",1)==0;
        if (ok) ok = mbedtls_md_hmac_update(&mac,(const uint8_t *)fields[i],strlen(fields[i]))==0;
    }
    uint8_t digest[32];
    if (ok) ok = mbedtls_md_hmac_finish(&mac,digest)==0;
    mbedtls_md_free(&mac);
    return ok && digest_matches(digest,string(root,"signature")) && freshness(fields[1],rfc3339(fields[0]));
}
static int compare_key(const void *a, const void *b) {
    return strcmp((*(const cJSON *const *)a)->string, (*(const cJSON *const *)b)->string);
}
static cJSON *canonical(const cJSON *v, unsigned depth) {
    if (!v || depth > 12) return NULL;
    if (!cJSON_IsObject(v) && !cJSON_IsArray(v)) return cJSON_Duplicate(v,true);
    int n = cJSON_GetArraySize(v);
    if (n > 128) return NULL;
    cJSON *out = cJSON_IsObject(v) ? cJSON_CreateObject() : cJSON_CreateArray();
    const cJSON **items = malloc((n ? n : 1) * sizeof(*items));
    if (!items || !out) { free(items); cJSON_Delete(out); return NULL; }
    for (int i=0; i<n; ++i) items[i]=cJSON_GetArrayItem(v,i);
    if (cJSON_IsObject(v)) qsort(items,n,sizeof(items[0]),compare_key);
    for (int i=0; i<n; ++i) {
        cJSON *child=canonical(items[i],depth+1);
        if (!child) { free(items); cJSON_Delete(out); return NULL; }
        if (cJSON_IsObject(v)) cJSON_AddItemToObject(out,items[i]->string,child);
        else cJSON_AddItemToArray(out,child);
    }
    free(items);
    return out;
}
static void be64(uint64_t value, uint8_t out[8]) {
    for (int i=7; i>=0; --i) { out[i]=value&255; value>>=8; }
}
bool nyx_security_verify_upload(cJSON *root, const char *secret, const char *node_id) {
    if (!root || !secret || !*secret || !node_id || strcmp(string(root,"node_id"),node_id) ||
        cJSON_GetNumberValue(cJSON_GetObjectItemCaseSensitive(root,"version")) != 1 ||
        strcmp(string(root,"operation"),"proxy_upload")) return false;
    uint8_t key[32];
    if (!secret_bytes(secret,key)) return false;
    const cJSON *timestamp=cJSON_GetObjectItemCaseSensitive(root,"timestamp");
    double raw_stamp=cJSON_GetNumberValue(timestamp);
    // Validate before converting: NaN, fractions and out-of-range casts are invalid.
    if (!cJSON_IsNumber(timestamp) || !isfinite(raw_stamp) || raw_stamp < 1700000000.0 ||
        raw_stamp > 4102444800.0 || raw_stamp != (double)(int64_t)raw_stamp) return false;
    cJSON *auth=cJSON_GetObjectItemCaseSensitive(root,"authority");
    if (auth && !cJSON_IsNull(auth)) return false;
    cJSON *params=canonical(cJSON_GetObjectItemCaseSensitive(root,"parameters"),0);
    if (!params) return false;
    char *json=cJSON_PrintUnformatted(params);
    cJSON_Delete(params);
    if (!json) return false;
    uint8_t sha[32], time_bytes[8], length[8], digest[32];
    mbedtls_sha256((const uint8_t *)json,strlen(json),sha,0);
    cJSON_free(json);
    int64_t stamp = (int64_t)raw_stamp;
    be64((uint64_t)stamp,time_bytes);
    const char *id=string(root,"request_id"), *nonce=string(root,"nonce");
    const uint8_t *fields[]={(const uint8_t *)id,(const uint8_t *)node_id,
        (const uint8_t *)"\"proxy_upload\"",sha,time_bytes,(const uint8_t *)nonce};
    size_t sizes[]={strlen(id),strlen(node_id),14,32,8,strlen(nonce)};
    mbedtls_md_context_t mac;
    mbedtls_md_init(&mac);
    bool ok=mbedtls_md_setup(&mac,mbedtls_md_info_from_type(MBEDTLS_MD_SHA256),1)==0 &&
        mbedtls_md_hmac_starts(&mac,key,sizeof(key))==0;
    const char domain[]="nyxid.machine.request.v1";
    if (ok) ok=mbedtls_md_hmac_update(&mac,(const uint8_t *)domain,sizeof(domain))==0;
    for (int i=0; ok && i<6; ++i) {
        be64(sizes[i],length);
        ok=mbedtls_md_hmac_update(&mac,length,8)==0 && mbedtls_md_hmac_update(&mac,fields[i],sizes[i])==0;
    }
    if (ok) ok=mbedtls_md_hmac_finish(&mac,digest)==0;
    mbedtls_md_free(&mac);
    return ok && strlen(id)==36 && digest_matches(digest,string(root,"signature")) && freshness(nonce,stamp);
}
