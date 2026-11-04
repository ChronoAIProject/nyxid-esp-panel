#include "nyx_creds.h"
#include "settings.h"

#include "devices.h"
#include "esp_log.h"
#include "nvs.h"

#include <stdio.h>
#include <string.h>

static const char *TAG = "haCam.creds";

static char s_api_key[128];
static char s_http_api_key[128];
static char s_node_id[96];
static char s_signing_secret[80];
static char s_slug[80];

bool nyx_creds_read_binding(nyx_binding_t *binding)
{
    if (!binding) return false;
    nvs_handle_t h;
    size_t len = sizeof(*binding);
    if (nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) return false;
    esp_err_t err = nvs_get_blob(h, "binding_v1", binding, &len);
    nvs_close(h);
    return err == ESP_OK && len == sizeof(*binding) &&
        memchr(binding->auth_token, 0, sizeof(binding->auth_token)) &&
        memchr(binding->node_id, 0, sizeof(binding->node_id)) &&
        memchr(binding->signing_secret, 0, sizeof(binding->signing_secret)) &&
        memchr(binding->slug, 0, sizeof(binding->slug));
}

static void apply_binding(const nyx_binding_t *binding)
{
    memcpy(s_api_key, binding->auth_token, sizeof(s_api_key));
    memcpy(s_http_api_key, binding->auth_token, sizeof(s_http_api_key));
    memcpy(s_node_id, binding->node_id, sizeof(s_node_id));
    memcpy(s_signing_secret, binding->signing_secret, sizeof(s_signing_secret));
    memcpy(s_slug, binding->slug, sizeof(s_slug));
}

int nyx_creds_store_binding(const nyx_binding_t *binding)
{
    if (!binding || !memchr(binding->auth_token, 0, sizeof(binding->auth_token)) ||
        !memchr(binding->node_id, 0, sizeof(binding->node_id)) ||
        !memchr(binding->signing_secret, 0, sizeof(binding->signing_secret)) ||
        !memchr(binding->slug, 0, sizeof(binding->slug))) return -1;
    nvs_handle_t h;
    if (nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) return -1;
    esp_err_t err = nvs_set_blob(h, "binding_v1", binding, sizeof(*binding));
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err == ESP_OK ? 0 : -1;
}

// Legacy UART/QR setters also update an identity created by the new login flow.
static int update_binding(const char *field, const char *value)
{
    nyx_binding_t binding = {0};
    if (!nyx_creds_read_binding(&binding)) return 1;
    char *dst = NULL;
    size_t cap = 0;
    if (!strcmp(field, "api_key")) { dst = binding.auth_token; cap = sizeof(binding.auth_token); }
    else if (!strcmp(field, "node_id")) { dst = binding.node_id; cap = sizeof(binding.node_id); }
    else if (!strcmp(field, "signing_secret")) { dst = binding.signing_secret; cap = sizeof(binding.signing_secret); }
    else if (!strcmp(field, "slug")) { dst = binding.slug; cap = sizeof(binding.slug); }
    if (!dst || strlen(value) >= cap) return -1;
    snprintf(dst, cap, "%s", value);
    int result = nyx_creds_store_binding(&binding);
    if (!result) apply_binding(&binding);
    return result;
}

static const char *baked(const char *s)
{
    return s ? s : "";
}

static void copy_string(char *dst, size_t cap, const char *src)
{
    if (cap == 0) {
        return;
    }
    if (!src) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, cap, "%s", src);
}

static void load_or_default(nvs_handle_t h, const char *key, char *dst, size_t cap,
                            const char *fallback)
{
    size_t len = cap;
    if (nvs_get_str(h, key, dst, &len) == ESP_OK) {
        return;
    }
    copy_string(dst, cap, fallback);
}

static bool creds_usable(void)
{
    return s_api_key[0] != '\0' &&
           s_http_api_key[0] != '\0' &&
           s_node_id[0] != '\0';
}

static bool strings_usable(const char *api_key, const char *http_api_key, const char *node_id)
{
    return api_key && http_api_key && node_id &&
           api_key[0] != '\0' &&
           http_api_key[0] != '\0' &&
           node_id[0] != '\0';
}

bool nyx_creds_have_nvs_credentials(void)
{
    nyx_binding_t binding = {0};
    if (nyx_creds_read_binding(&binding)) return binding.auth_token[0] && binding.node_id[0];
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READONLY, &h);
    if (err != ESP_OK) {
        return false;
    }

    char api_key[128] = "";
    char http_api_key[128] = "";
    char node_id[96] = "";
    size_t len = sizeof(api_key);
    bool have = nvs_get_str(h, "api_key", api_key, &len) == ESP_OK;
    len = sizeof(http_api_key);
    if (nvs_get_str(h, "http_api_key", http_api_key, &len) != ESP_OK) {
        copy_string(http_api_key, sizeof(http_api_key), api_key);
    }
    len = sizeof(node_id);
    have = have && nvs_get_str(h, "node_id", node_id, &len) == ESP_OK;
    nvs_close(h);

    return have && strings_usable(api_key, http_api_key, node_id);
}

bool nyx_creds_load(void)
{
    nyx_binding_t binding = {0};
    if (nyx_creds_read_binding(&binding)) {
        apply_binding(&binding);
        return creds_usable();
    }
    const device_config_t *self = NYX_NODE_ALLOW_BAKED_CREDENTIALS ? devices_self() : NULL;
    const char *fallback_api      = self ? baked(self->nyx_api_key)        : "";
    const char *fallback_http     = self ? baked(self->nyx_api_key_http)   : "";
    const char *fallback_node     = self ? baked(self->nyx_node_id)        : "";
    const char *fallback_secret   = self ? baked(self->nyx_signing_secret) : "";
    const char *fallback_slug     = self ? baked(self->slug)               : "";

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Nyx credentials unset; runtime provisioning is available");
        copy_string(s_api_key,        sizeof(s_api_key),        fallback_api);
        copy_string(s_http_api_key,   sizeof(s_http_api_key),   fallback_http);
        copy_string(s_node_id,        sizeof(s_node_id),        fallback_node);
        copy_string(s_signing_secret, sizeof(s_signing_secret), fallback_secret);
        copy_string(s_slug,           sizeof(s_slug),           fallback_slug);
        return creds_usable();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_open(nyx) failed: %s", esp_err_to_name(err));
        return false;
    }

    load_or_default(h, "api_key",        s_api_key,        sizeof(s_api_key),        fallback_api);
    load_or_default(h, "http_api_key",   s_http_api_key,   sizeof(s_http_api_key),   fallback_http);
    load_or_default(h, "node_id",        s_node_id,        sizeof(s_node_id),        fallback_node);
    load_or_default(h, "signing_secret", s_signing_secret, sizeof(s_signing_secret), fallback_secret);
    load_or_default(h, "slug",           s_slug,           sizeof(s_slug),           fallback_slug);
    nvs_close(h);

    return creds_usable();
}

const char *nyx_creds_api_key(void)
{
    return s_api_key;
}

const char *nyx_creds_http_api_key(void)
{
    return s_http_api_key;
}

const char *nyx_creds_node_id(void)
{
    return s_node_id;
}

const char *nyx_creds_signing_secret(void)
{
    return s_signing_secret;
}

const char *nyx_creds_service_slug(void)
{
    return s_slug;
}

int nyx_creds_save_api_key(const char *api_key)
{
    if (!api_key || api_key[0] == '\0') {
        return -1;
    }
    int updated = update_binding("api_key", api_key);
    if (updated != 1) return updated;
    ESP_LOGI(TAG, "write namespace=nyx key=api_key value=<redacted> len=%u", (unsigned)strlen(api_key));

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_open(nyx) failed while saving api key: %s", esp_err_to_name(err));
        return -1;
    }

    err = nvs_set_str(h, "api_key", api_key);
    if (err == ESP_OK) {
        err = nvs_set_str(h, "http_api_key", api_key);
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "api key save failed: %s", esp_err_to_name(err));
        return -1;
    }

    copy_string(s_api_key, sizeof(s_api_key), api_key);
    copy_string(s_http_api_key, sizeof(s_http_api_key), api_key);
    return 0;
}

int nyx_creds_save_node_id(const char *node_id)
{
    if (!node_id || node_id[0] == '\0') {
        return -1;
    }
    int updated = update_binding("node_id", node_id);
    if (updated != 1) return updated;
    ESP_LOGI(TAG, "write namespace=nyx key=node_id value=%s", node_id);

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_open(nyx) failed while saving node id: %s", esp_err_to_name(err));
        return -1;
    }

    err = nvs_set_str(h, "node_id", node_id);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "node id save failed: %s", esp_err_to_name(err));
        return -1;
    }

    copy_string(s_node_id, sizeof(s_node_id), node_id);
    return 0;
}

int nyx_creds_save_signing_secret(const char *secret)
{
    if (!secret || secret[0] == '\0') {
        return -1;
    }
    int updated = update_binding("signing_secret", secret);
    if (updated != 1) return updated;
    ESP_LOGI(TAG, "write namespace=nyx key=signing_secret value=<redacted> len=%u",
             (unsigned)strlen(secret));

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_open(nyx) failed while saving signing secret: %s", esp_err_to_name(err));
        return -1;
    }

    err = nvs_set_str(h, "signing_secret", secret);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "signing secret save failed: %s", esp_err_to_name(err));
        return -1;
    }

    copy_string(s_signing_secret, sizeof(s_signing_secret), secret);
    return 0;
}

int nyx_creds_save_slug(const char *slug)
{
    if (!slug || slug[0] == '\0') {
        return -1;
    }
    int updated = update_binding("slug", slug);
    if (updated != 1) return updated;
    ESP_LOGI(TAG, "write namespace=nyx key=slug value=%s", slug);

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open(NYX_NODE_NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs_open(nyx) failed while saving slug: %s", esp_err_to_name(err));
        return -1;
    }

    err = nvs_set_str(h, "slug", slug);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "slug save failed: %s", esp_err_to_name(err));
        return -1;
    }

    copy_string(s_slug, sizeof(s_slug), slug);
    return 0;
}

static bool agent_valid(const nyx_agent_credential_t *value)
{
    if (!value || !memchr(value->credential,0,sizeof(value->credential)) ||
        !memchr(value->credential_id,0,sizeof(value->credential_id)) ||
        !memchr(value->key_id,0,sizeof(value->key_id)) ||
        !memchr(value->owner_id,0,sizeof(value->owner_id)) ||
        !memchr(value->label,0,sizeof(value->label))) return false;
    // Login always delivers a child Agent Key, even when the selected parent
    // is a primary key. Never accept a JWT or a node token here.
    if (strlen(value->credential)!=73 || strncmp(value->credential,"nyxid_ag_",9) ||
        strlen(value->credential_id)!=36 || strlen(value->key_id)!=36 || strlen(value->owner_id)!=36) return false;
    for (const char *p=value->credential+9;*p;p++) if (!strchr("0123456789abcdef",*p)) return false;
    return true;
}
bool nyx_creds_read_agent(nyx_agent_credential_t *out)
{
    if (!out) return false;
    memset(out,0,sizeof(*out));
    nvs_handle_t h;
    if (nvs_open(NYX_AGENT_NVS_NAMESPACE,NVS_READONLY,&h)!=ESP_OK) return false;
    size_t len=sizeof(*out);esp_err_t err=nvs_get_blob(h,"credential_v1",out,&len);nvs_close(h);
    if (err!=ESP_OK || len!=sizeof(*out) || !agent_valid(out)) {memset(out,0,sizeof(*out));return false;}
    return true;
}
bool nyx_creds_have_agent(void)
{
    nyx_agent_credential_t value;
    bool valid = nyx_creds_read_agent(&value);
    memset(&value, 0, sizeof(value));
    return valid;
}
int nyx_creds_store_agent(const nyx_agent_credential_t *value)
{
    if (!agent_valid(value)) return -1;
    nvs_handle_t h;
    if (nvs_open(NYX_AGENT_NVS_NAMESPACE,NVS_READWRITE,&h)!=ESP_OK) return -1;
    esp_err_t err=nvs_set_blob(h,"credential_v1",value,sizeof(*value));
    if (err==ESP_OK) err=nvs_commit(h);
    nvs_close(h);return err==ESP_OK?0:-1;
}
