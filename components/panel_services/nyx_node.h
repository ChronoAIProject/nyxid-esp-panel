#pragma once

#include "nyx_proto.h"

#include <stdbool.h>
#include <stddef.h>

typedef int (*nyx_handler_fn)(const nyx_proxy_request_t *req,
                              char *resp_body, size_t resp_cap,
                              char *content_type, size_t ct_cap,
                              size_t *resp_len,
                              void *user);

typedef struct {
    const char *path;
    const char *method;
    nyx_handler_fn fn;
    void *user;
} nyx_route_t;

typedef int (*nyx_route_wrapper_fn)(const nyx_route_t *route,
                                    const nyx_proxy_request_t *req,
                                    char *resp_body, size_t resp_cap,
                                    char *content_type, size_t ct_cap,
                                    size_t *resp_len);

typedef enum {
    NYX_NODE_DOWN = 0,
    NYX_NODE_CONNECTING,
    NYX_NODE_UP,
} nyx_node_state_t;

void nyx_node_init(void);
void nyx_node_set_routes(const nyx_route_t *routes, size_t count);
void nyx_node_set_route_wrapper(nyx_route_wrapper_fn wrapper);
void nyx_node_start(void);
void nyx_node_on_network(bool connected);

nyx_node_state_t nyx_node_state(void);
const char *nyx_node_last_error(void);
