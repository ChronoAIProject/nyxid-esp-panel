#pragma once

#include <stdbool.h>

typedef struct {
    char auth_token[128];
    char node_id[96];
    char signing_secret[80];
    char slug[80];
} nyx_binding_t;

// One NVS blob replaces the complete identity atomically. Applied on reboot.
int nyx_creds_store_binding(const nyx_binding_t *binding);
bool nyx_creds_read_binding(nyx_binding_t *binding);

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

// Outbound Agent Key login is independent of the inbound node identity.
typedef struct {
    char credential[128];
    char credential_id[40];
    char key_id[40];
    char owner_id[40];
    char label[80];
} nyx_agent_credential_t;
bool nyx_creds_read_agent(nyx_agent_credential_t *out);
bool nyx_creds_have_agent(void);
int nyx_creds_store_agent(const nyx_agent_credential_t *value);
