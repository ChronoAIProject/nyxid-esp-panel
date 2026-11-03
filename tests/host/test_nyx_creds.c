#include "unity.h"
#include "nyx_creds.h"
#include "nvs_config.h"
#include "devices.h"
#include "nvs.h"
#include "nvs_flash.h"
#include <string.h>
#include <stdio.h>

// An SDK API substitute tests storage migration and reset behavior. It does not
// establish physical flash durability or replace real power-loss acceptance.
static char namespaces[16][32];
static struct { unsigned ns; char key[32]; unsigned char data[512]; size_t size; } values[64];
static bool fail_blob;

const device_config_t *devices_self(void) { return NULL; }
esp_err_t nvs_flash_init(void) { return ESP_OK; }
esp_err_t nvs_flash_deinit(void) { return ESP_OK; }
esp_err_t nvs_flash_erase(void)
{
    memset(namespaces, 0, sizeof(namespaces)); memset(values, 0, sizeof(values));
    return ESP_OK;
}
esp_err_t nvs_open(const char *name, int mode, nvs_handle_t *handle)
{
    for (unsigned i = 1; i < 16; ++i) if (!strcmp(namespaces[i], name)) { *handle = i; return ESP_OK; }
    if (mode == NVS_READONLY) return ESP_ERR_NVS_NOT_FOUND;
    for (unsigned i = 1; i < 16; ++i) if (!namespaces[i][0]) {
        snprintf(namespaces[i], sizeof(namespaces[i]), "%s", name); *handle = i; return ESP_OK;
    }
    return ESP_FAIL;
}
void nvs_close(nvs_handle_t handle) { (void)handle; }
esp_err_t nvs_commit(nvs_handle_t handle) { (void)handle; return ESP_OK; }
static esp_err_t put(nvs_handle_t handle, const char *key, const void *data, size_t size)
{
    unsigned slot = 64;
    for (unsigned i = 0; i < 64; ++i) {
        if (values[i].ns == handle && !strcmp(values[i].key, key)) { slot = i; break; }
        if (!values[i].ns && slot == 64) slot = i;
    }
    if (slot == 64 || size > sizeof(values[slot].data)) return ESP_FAIL;
    values[slot].ns = handle; values[slot].size = size;
    snprintf(values[slot].key, sizeof(values[slot].key), "%s", key);
    memcpy(values[slot].data, data, size);
    return ESP_OK;
}
static esp_err_t get(nvs_handle_t handle, const char *key, void *data, size_t *size)
{
    for (unsigned i = 0; i < 64; ++i) if (values[i].ns == handle && !strcmp(values[i].key, key)) {
        size_t available = *size; *size = values[i].size;
        if (!data) return ESP_OK;
        if (available < *size) return ESP_ERR_NVS_INVALID_LENGTH;
        memcpy(data, values[i].data, *size); return ESP_OK;
    }
    return ESP_ERR_NVS_NOT_FOUND;
}
esp_err_t nvs_set_blob(nvs_handle_t h, const char *key, const void *data, size_t size)
{
    if (fail_blob) { fail_blob = false; return ESP_FAIL; }
    return put(h, key, data, size);
}
esp_err_t nvs_get_blob(nvs_handle_t h, const char *key, void *data, size_t *size) { return get(h, key, data, size); }
esp_err_t nvs_set_str(nvs_handle_t h, const char *key, const char *value) { return put(h, key, value, strlen(value) + 1); }
esp_err_t nvs_get_str(nvs_handle_t h, const char *key, char *value, size_t *size) { return get(h, key, value, size); }
#define NUMBER(bits) \
esp_err_t nvs_set_u##bits(nvs_handle_t h, const char *key, uint##bits##_t value) { return put(h, key, &value, sizeof(value)); } \
esp_err_t nvs_get_u##bits(nvs_handle_t h, const char *key, uint##bits##_t *value) { size_t size = sizeof(*value); return get(h, key, value, &size); }
NUMBER(8)
NUMBER(16)
NUMBER(32)

void setUp(void) { nvs_flash_erase(); fail_blob = false; (void)nyx_creds_load(); }
void tearDown(void) {}
static nyx_binding_t binding(void)
{
    nyx_binding_t value = {0};
    strcpy(value.auth_token, "nyx_nauth_new");
    strcpy(value.node_id, "11111111-2222-3333-4444-555555555555");
    memset(value.signing_secret, 'a', 64);
    strcpy(value.slug, "panel-new");
    return value;
}
static void legacy(void)
{
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_save_api_key("legacy-node-token"));
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_save_node_id("legacy-node"));
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_save_slug("legacy-service"));
}

void test_identity_activates_on_reload_and_survives_preserving_reset(void)
{
    legacy();
    nyx_binding_t next = binding();
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_store_binding(&next));
    TEST_ASSERT_EQUAL_STRING("legacy-node-token", nyx_creds_api_key());
    TEST_ASSERT_TRUE(nyx_creds_load());
    TEST_ASSERT_EQUAL_STRING(next.auth_token, nyx_creds_api_key());
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_factory_reset_preserve());
    TEST_ASSERT_TRUE(nyx_creds_load());
    TEST_ASSERT_EQUAL_STRING(next.node_id, nyx_creds_node_id());
    TEST_ASSERT_EQUAL_STRING(next.slug, nyx_creds_service_slug());
    TEST_ASSERT_EQUAL_STRING(next.signing_secret, nyx_creds_signing_secret());
}

void test_failed_write_retains_legacy_and_uart_updates_new_format(void)
{
    legacy();
    nyx_binding_t next = binding();
    fail_blob = true;
    TEST_ASSERT_EQUAL_INT(-1, nyx_creds_store_binding(&next));
    TEST_ASSERT_TRUE(nyx_creds_load());
    TEST_ASSERT_EQUAL_STRING("legacy-node-token", nyx_creds_api_key());
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_store_binding(&next));
    TEST_ASSERT_TRUE(nyx_creds_load());
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_save_slug("panel-renamed"));
    TEST_ASSERT_TRUE(nyx_creds_load());
    TEST_ASSERT_EQUAL_STRING("panel-renamed", nyx_creds_service_slug());
    TEST_ASSERT_EQUAL_STRING(next.auth_token, nyx_creds_api_key());
    fail_blob = true;
    TEST_ASSERT_EQUAL_INT(-1, nyx_creds_save_api_key("replacement-token"));
    TEST_ASSERT_EQUAL_STRING(next.auth_token, nyx_creds_api_key());
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_identity_activates_on_reload_and_survives_preserving_reset);
    RUN_TEST(test_failed_write_retains_legacy_and_uart_updates_new_format);
    return UNITY_END();
}
