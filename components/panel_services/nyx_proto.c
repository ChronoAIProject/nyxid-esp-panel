#include "nyx_proto.h"

#include "b64.h"
#include "cjson_tiny.h"
#include "nyx_creds.h"
#include "settings.h"

#if HA_CAM_HMAC_SIGN_FRAMES
#include "esp_random.h"
#include "esp_timer.h"
#include "mbedtls/md.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#if HA_CAM_HMAC_SIGN_FRAMES
static int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool hex_to_bytes(const char *hex, uint8_t *out, size_t out_cap, size_t *out_len)
{
    size_t len = hex ? strlen(hex) : 0u;
    if (!hex || (len % 2u) != 0u || (len / 2u) > out_cap || !out_len) {
        return false;
    }
    for (size_t i = 0; i < len; i += 2u) {
        int hi = hex_value(hex[i]);
        int lo = hex_value(hex[i + 1u]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out[i / 2u] = (uint8_t)((hi << 4) | lo);
    }
    *out_len = len / 2u;
    return true;
}

static void bytes_to_hex(const uint8_t *bytes, size_t len, char out_hex[65])
{
    static const char chars[] = "0123456789abcdef";
    for (size_t i = 0; i < len; ++i) {
        out_hex[i * 2u] = chars[(bytes[i] >> 4) & 0x0f];
        out_hex[i * 2u + 1u] = chars[bytes[i] & 0x0f];
    }
    out_hex[len * 2u] = '\0';
}

static bool hmac_sha256_hex(const char *key, size_t key_len,
                            const char *msg, size_t msg_len,
                            char out_hex[65])
{
    uint8_t key_bytes[32] = {0};
    size_t decoded_len = 0u;
    if (key_len != 64u || !hex_to_bytes(key, key_bytes, sizeof(key_bytes), &decoded_len)) {
        return false;
    }

    const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    uint8_t mac[32] = {0};
    if (!info ||
        mbedtls_md_hmac(info, key_bytes, decoded_len,
                        (const unsigned char *)msg, msg_len, mac) != 0) {
        return false;
    }
    bytes_to_hex(mac, sizeof(mac), out_hex);
    return true;
}

static void make_signing_fields(char *timestamp, size_t timestamp_cap,
                                char *nonce, size_t nonce_cap)
{
    snprintf(timestamp, timestamp_cap, "%lld", (long long)esp_timer_get_time());
    snprintf(nonce, nonce_cap, "%08lx%08lx",
             (unsigned long)esp_random(), (unsigned long)esp_random());
}

static void maybe_sign_auth(cJSON *obj, const char *node_id, const char *token)
{
    const char *secret = nyx_creds_signing_secret();
    if (!secret || secret[0] == '\0' || strcmp(secret, "REPLACE_ME") == 0) {
        return;
    }

    char timestamp[32];
    char nonce[24];
    char msg[512];
    char sig[65];
    make_signing_fields(timestamp, sizeof(timestamp), nonce, sizeof(nonce));
    int n = snprintf(msg, sizeof(msg), "%s\n%s\n%s\n%s", timestamp, nonce, node_id, token);
    if (n <= 0 || (size_t)n >= sizeof(msg) ||
        !hmac_sha256_hex(secret, strlen(secret), msg, (size_t)n, sig)) {
        return;
    }
    cJSON_AddStringToObject(obj, "timestamp", timestamp);
    cJSON_AddStringToObject(obj, "nonce", nonce);
    cJSON_AddStringToObject(obj, "signature", sig);
}

static void maybe_sign_proxy_response(cJSON *obj, const char *request_id, int status,
                                      const char *body_b64)
{
    const char *secret = nyx_creds_signing_secret();
    if (!secret || secret[0] == '\0' || strcmp(secret, "REPLACE_ME") == 0) {
        return;
    }

    char timestamp[32];
    char nonce[24];
    char msg[768];
    char sig[65];
    make_signing_fields(timestamp, sizeof(timestamp), nonce, sizeof(nonce));
    int n = snprintf(msg, sizeof(msg), "%s\n%s\n%s\n%d\n%s",
                     timestamp, nonce, request_id, status, body_b64 ? body_b64 : "");
    if (n <= 0 || (size_t)n >= sizeof(msg) ||
        !hmac_sha256_hex(secret, strlen(secret), msg, (size_t)n, sig)) {
        return;
    }
    cJSON_AddStringToObject(obj, "timestamp", timestamp);
    cJSON_AddStringToObject(obj, "nonce", nonce);
    cJSON_AddStringToObject(obj, "signature", sig);
}
#else
static void maybe_sign_auth(cJSON *obj, const char *node_id, const char *token)
{
    (void)obj;
    (void)node_id;
    (void)token;
}

static void maybe_sign_proxy_response(cJSON *obj, const char *request_id, int status,
                                      const char *body_b64)
{
    (void)obj;
    (void)request_id;
    (void)status;
    (void)body_b64;
}
#endif

static char *dup_json_string(cJSON *obj, const char *name)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, name);
    if (!cJSON_IsString(item) || item->valuestring == NULL) {
        return NULL;
    }

    size_t len = strlen(item->valuestring);
    char *copy = (char *)malloc(len + 1u);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, item->valuestring, len + 1u);
    return copy;
}

static nyx_msg_kind_t kind_from_type(const char *type)
{
    if (!type) {
        return NYX_MSG_UNKNOWN;
    }
    if (strcmp(type, "register_ok") == 0) {
        return NYX_MSG_REGISTER_OK;
    }
    if (strcmp(type, "auth_ok") == 0) {
        return NYX_MSG_AUTH_OK;
    }
    if (strcmp(type, "auth_error") == 0) {
        return NYX_MSG_AUTH_ERROR;
    }
    if (strcmp(type, "heartbeat_ping") == 0) {
        return NYX_MSG_HEARTBEAT_PING;
    }
    if (strcmp(type, "proxy_request") == 0) {
        return NYX_MSG_PROXY_REQUEST;
    }
    return NYX_MSG_UNKNOWN;
}

static void parse_headers(cJSON *root, nyx_proxy_request_t *req)
{
    cJSON *headers = cJSON_GetObjectItemCaseSensitive(root, "headers");
    if (!cJSON_IsObject(headers)) {
        return;
    }

    cJSON *item = NULL;
    cJSON_ArrayForEach(item, headers) {
        if (req->header_count >= NYX_PROTO_MAX_HEADERS) {
            break;
        }
        if (!cJSON_IsString(item) || item->string == NULL || item->valuestring == NULL) {
            continue;
        }
        size_t idx = req->header_count;
        req->headers[idx].name = (char *)malloc(strlen(item->string) + 1u);
        req->headers[idx].value = (char *)malloc(strlen(item->valuestring) + 1u);
        if (!req->headers[idx].name || !req->headers[idx].value) {
            free(req->headers[idx].name);
            free(req->headers[idx].value);
            req->headers[idx].name = NULL;
            req->headers[idx].value = NULL;
            continue;
        }
        strcpy(req->headers[idx].name, item->string);
        strcpy(req->headers[idx].value, item->valuestring);
        req->header_count++;
    }
}

static int parse_body(cJSON *root, nyx_proxy_request_t *req)
{
    cJSON *body = cJSON_GetObjectItemCaseSensitive(root, "body");
    if (!cJSON_IsString(body) || body->valuestring == NULL) {
        return 0;
    }

    size_t in_len = strlen(body->valuestring);
    size_t out_cap = b64_decoded_max_len(in_len);
    uint8_t *decoded = (uint8_t *)malloc(out_cap == 0u ? 1u : out_cap);
    if (!decoded) {
        return -1;
    }

    size_t out_len = 0;
    if (b64_decode(body->valuestring, in_len, decoded, out_cap, &out_len) != 0) {
        free(decoded);
        return -1;
    }

    req->body = decoded;
    req->body_len = out_len;
    return 0;
}

static void parse_proxy_request(cJSON *root, nyx_proxy_request_t *req)
{
    memset(req, 0, sizeof(*req));
    req->request_id = dup_json_string(root, "request_id");
    req->service_id = dup_json_string(root, "service_id");
    req->service_slug = dup_json_string(root, "service_slug");
    req->base_url = dup_json_string(root, "base_url");
    req->method = dup_json_string(root, "method");
    req->path = dup_json_string(root, "path");
    req->query = dup_json_string(root, "query");
    req->timestamp = dup_json_string(root, "timestamp");
    req->nonce = dup_json_string(root, "nonce");
    req->signature = dup_json_string(root, "signature");
    parse_headers(root, req);
    if (parse_body(root, req) != 0) {
        nyx_proto_free_request(req);
    }
}

static void parse_auth_result(cJSON *root, nyx_auth_result_t *auth)
{
    memset(auth, 0, sizeof(*auth));
    auth->node_id = dup_json_string(root, "node_id");
    auth->auth_token = dup_json_string(root, "auth_token");
    auth->signing_secret = dup_json_string(root, "signing_secret");
    auth->message = dup_json_string(root, "message");
    auth->timestamp = dup_json_string(root, "timestamp");
    cJSON *heartbeat = cJSON_GetObjectItemCaseSensitive(root, "heartbeat_interval_secs");
    if (cJSON_IsNumber(heartbeat) && heartbeat->valuedouble > 0.0 &&
        heartbeat->valuedouble <= UINT32_MAX) {
        auth->heartbeat_interval_secs = (uint32_t)heartbeat->valuedouble;
    }
}

nyx_msg_kind_t nyx_proto_parse_text(const char *text, nyx_proxy_request_t *req,
                                    nyx_auth_result_t *auth)
{
    if (!text) {
        return NYX_MSG_UNKNOWN;
    }

    cJSON *root = cJSON_Parse(text);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return NYX_MSG_UNKNOWN;
    }

    cJSON *type_item = cJSON_GetObjectItemCaseSensitive(root, "type");
    nyx_msg_kind_t kind = kind_from_type(cJSON_IsString(type_item) ? type_item->valuestring : NULL);

    if (kind == NYX_MSG_PROXY_REQUEST && req) {
        parse_proxy_request(root, req);
    }
    if ((kind == NYX_MSG_REGISTER_OK || kind == NYX_MSG_AUTH_OK ||
         kind == NYX_MSG_AUTH_ERROR || kind == NYX_MSG_HEARTBEAT_PING) &&
        auth) {
        parse_auth_result(root, auth);
    }

    cJSON_Delete(root);
    return kind;
}

void nyx_proto_free_request(nyx_proxy_request_t *req)
{
    if (!req) {
        return;
    }
    free(req->request_id);
    free(req->service_id);
    free(req->service_slug);
    free(req->base_url);
    free(req->method);
    free(req->path);
    free(req->query);
    for (size_t i = 0; i < req->header_count; ++i) {
        free(req->headers[i].name);
        free(req->headers[i].value);
    }
    free(req->body);
    free(req->timestamp);
    free(req->nonce);
    free(req->signature);
    memset(req, 0, sizeof(*req));
}

void nyx_proto_free_auth_result(nyx_auth_result_t *auth)
{
    if (!auth) {
        return;
    }
    free(auth->node_id);
    free(auth->auth_token);
    free(auth->signing_secret);
    free(auth->message);
    free(auth->timestamp);
    memset(auth, 0, sizeof(*auth));
}

static size_t print_json(cJSON *obj, char *out, size_t cap)
{
    char *printed = cJSON_PrintUnformatted(obj);
    if (!printed) {
        return 0u;
    }

    size_t len = strlen(printed);
    if (len + 1u > cap) {
        cJSON_free(printed);
        return 0u;
    }
    memcpy(out, printed, len + 1u);
    cJSON_free(printed);
    return len;
}

size_t nyx_proto_build_register_token(char *out, size_t cap, const char *token)
{
    if (!out || !token) {
        return 0u;
    }
    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "register");
    cJSON_AddStringToObject(obj, "token", token);
    size_t len = print_json(obj, out, cap);
    cJSON_Delete(obj);
    return len;
}

size_t nyx_proto_build_auth(char *out, size_t cap, const char *node_id,
                            const char *token)
{
    if (!out || !node_id || !token) {
        return 0u;
    }
    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "auth");
    cJSON_AddStringToObject(obj, "node_id", node_id);
    cJSON_AddStringToObject(obj, "token", token);
    maybe_sign_auth(obj, node_id, token);
    size_t len = print_json(obj, out, cap);
    cJSON_Delete(obj);
    return len;
}

size_t nyx_proto_build_proxy_response(char *out, size_t cap,
                                      const char *request_id, int status,
                                      const char *content_type,
                                      const void *body, size_t body_len)
{
    if (!out || !request_id) {
        return 0u;
    }

    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "proxy_response");
    cJSON_AddStringToObject(obj, "request_id", request_id);
    cJSON_AddNumberToObject(obj, "status", status);
    char *body_b64 = NULL;

    cJSON *headers = cJSON_AddObjectToObject(obj, "headers");
    if (content_type) {
        cJSON_AddStringToObject(headers, "content-type", content_type);
    }

    if (body && body_len > 0u) {
        size_t enc_cap = b64_encoded_len(body_len) + 1u;
        char *enc = (char *)malloc(enc_cap);
        size_t enc_len = 0;
        if (!enc || b64_encode((const uint8_t *)body, body_len, enc, enc_cap, &enc_len) != 0) {
            free(enc);
            cJSON_Delete(obj);
            return 0u;
        }
        cJSON_AddStringToObject(obj, "body", enc);
        body_b64 = enc;
    }

    maybe_sign_proxy_response(obj, request_id, status, body_b64);
    size_t len = print_json(obj, out, cap);
    free(body_b64);
    cJSON_Delete(obj);
    return len;
}

size_t nyx_proto_build_pong(char *out, size_t cap, const char *timestamp)
{
    if (!out) {
        return 0u;
    }
    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "heartbeat_pong");
    if (timestamp) {
        cJSON_AddStringToObject(obj, "timestamp", timestamp);
    }
    size_t len = print_json(obj, out, cap);
    cJSON_Delete(obj);
    return len;
}
