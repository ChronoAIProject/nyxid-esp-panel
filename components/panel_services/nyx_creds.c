#include "nyx_creds.h"

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
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("nyx", NVS_READONLY, &h);
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
    const device_config_t *self = devices_self();
    const char *fallback_api      = self ? baked(self->nyx_api_key)        : "";
    const char *fallback_http     = self ? baked(self->nyx_api_key_http)   : "";
    const char *fallback_node     = self ? baked(self->nyx_node_id)        : "";
    const char *fallback_secret   = self ? baked(self->nyx_signing_secret) : "";
    const char *fallback_slug     = self ? baked(self->slug)               : "";

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("nyx", NVS_READONLY, &h);
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
    ESP_LOGI(TAG, "write namespace=nyx key=api_key value=<redacted> len=%u", (unsigned)strlen(api_key));

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("nyx", NVS_READWRITE, &h);
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
    ESP_LOGI(TAG, "write namespace=nyx key=node_id value=%s", node_id);

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("nyx", NVS_READWRITE, &h);
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
    ESP_LOGI(TAG, "write namespace=nyx key=signing_secret value=<redacted> len=%u",
             (unsigned)strlen(secret));

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("nyx", NVS_READWRITE, &h);
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
    ESP_LOGI(TAG, "write namespace=nyx key=slug value=%s", slug);

    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("nyx", NVS_READWRITE, &h);
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
