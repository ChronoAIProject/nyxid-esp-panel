#pragma once

#include <stddef.h>
#include <stdint.h>

#define NYX_PROTO_MAX_HEADERS 16u

typedef enum {
    NYX_MSG_UNKNOWN = 0,
    NYX_MSG_REGISTER_OK,
    NYX_MSG_AUTH_OK,
    NYX_MSG_AUTH_ERROR,
    NYX_MSG_HEARTBEAT_PING,
    NYX_MSG_PROXY_REQUEST,
} nyx_msg_kind_t;

typedef struct {
    char *name;
    char *value;
} nyx_header_t;

typedef struct {
    char *request_id;
    char *service_id;
    char *service_slug;
    char *base_url;
    char *method;
    char *path;
    char *query;
    nyx_header_t headers[NYX_PROTO_MAX_HEADERS];
    size_t header_count;
    uint8_t *body;
    size_t body_len;
    char *timestamp;
    char *nonce;
    char *signature;
} nyx_proxy_request_t;

typedef struct {
    char *node_id;
    char *auth_token;
    char *signing_secret;
    char *message;
    char *timestamp;
    uint32_t heartbeat_interval_secs;
} nyx_auth_result_t;

nyx_msg_kind_t nyx_proto_parse_text(const char *text, nyx_proxy_request_t *req,
                                    nyx_auth_result_t *auth);
void nyx_proto_free_request(nyx_proxy_request_t *req);
void nyx_proto_free_auth_result(nyx_auth_result_t *auth);

size_t nyx_proto_build_register_token(char *out, size_t cap, const char *token);
size_t nyx_proto_build_auth(char *out, size_t cap, const char *node_id,
                            const char *token);
size_t nyx_proto_build_proxy_response(char *out, size_t cap,
                                      const char *request_id, int status,
                                      const char *content_type,
                                      const void *body, size_t body_len);
size_t nyx_proto_build_pong(char *out, size_t cap, const char *timestamp);
