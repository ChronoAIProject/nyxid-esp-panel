#pragma once

#include <stdbool.h>

bool nyx_creds_load(void);
bool nyx_creds_have_nvs_credentials(void);
const char *nyx_creds_api_key(void);
const char *nyx_creds_http_api_key(void);
const char *nyx_creds_node_id(void);
const char *nyx_creds_signing_secret(void);
const char *nyx_creds_service_slug(void);
int nyx_creds_save_api_key(const char *api_key);
int nyx_creds_save_node_id(const char *node_id);
int nyx_creds_save_signing_secret(const char *secret);
int nyx_creds_save_slug(const char *slug);
