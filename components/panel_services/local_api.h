#pragma once

#include "nyx_node.h"

void local_api_init(void);
const nyx_route_t *local_api_routes(size_t *count);
const nyx_route_t *local_api_lan_routes(size_t *count);
int local_api_route_with_logging(const nyx_route_t *route,
                                 const nyx_proxy_request_t *req,
                                 char *resp_body, size_t resp_cap,
                                 char *content_type, size_t ct_cap,
                                 size_t *resp_len);
size_t local_api_format_status_json(char *resp, size_t cap);
