#include "unity.h"
#include "nyx_creds.h"
#include "nvs_config.h"
#include "devices.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "settings.h"
#include <string.h>
#include <stdio.h>

// An SDK API substitute tests storage migration and reset behavior. It does not
// establish physical flash durability or replace real power-loss acceptance.
static char namespaces[16][32];
static struct { unsigned ns; char key[32]; unsigned char data[512]; size_t size; } values[64];
static bool fail_blob, fail_erase;

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
esp_err_t nvs_erase_all(nvs_handle_t handle)
{
    if (fail_erase) { fail_erase = false; return ESP_FAIL; }
    for (unsigned i = 0; i < 64; ++i)
        if (values[i].ns == handle) memset(&values[i], 0, sizeof(values[i]));
    return ESP_OK;
}
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

void setUp(void) { nvs_flash_erase(); fail_blob = fail_erase = false; (void)nyx_creds_load(); }
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

void test_agent_key_is_independent_atomic_and_preserved_with_identity(void)
{
    nyx_binding_t node = binding();
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_store_binding(&node));
    nyx_agent_credential_t agent = {0}, restored = {0};
    strcpy(agent.credential, "nyxid_ag_"); memset(agent.credential + 9, 'a', 64);
    strcpy(agent.credential_id, "11111111-2222-3333-4444-555555555555");
    strcpy(agent.key_id, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
    strcpy(agent.owner_id, "aaaaaaaa-1111-2222-3333-444444444444");
    strcpy(agent.label, "Panel at home");
    TEST_ASSERT_FALSE(nyx_creds_have_agent());
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_store_agent(&agent));
    TEST_ASSERT_TRUE(nyx_creds_read_agent(&restored));
    TEST_ASSERT_EQUAL_STRING(agent.credential, restored.credential);
    fail_blob = true;
    agent.credential[9] = 'b';
    TEST_ASSERT_EQUAL_INT(-1, nyx_creds_store_agent(&agent));
    TEST_ASSERT_TRUE(nyx_creds_read_agent(&restored));
    TEST_ASSERT_EQUAL_INT('a', restored.credential[9]);
    strcpy(agent.credential, "temporary-account-session");
    TEST_ASSERT_EQUAL_INT(-1, nyx_creds_store_agent(&agent));
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_factory_reset_preserve());
    TEST_ASSERT_TRUE(nyx_creds_read_agent(&agent));
    TEST_ASSERT_EQUAL_STRING(restored.credential, agent.credential);
    TEST_ASSERT_TRUE(nyx_creds_read_binding(&node));
    TEST_ASSERT_EQUAL_STRING("nyx_nauth_new", node.auth_token);
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_factory_reset_all());
    TEST_ASSERT_FALSE(nyx_creds_have_agent());
}

void test_preserving_reset_keeps_inactive_profiles_wifi_and_pending_setup(void)
{
    const char *const profiles[] = {
        "nyx", "nyx_agent", "nyx_setup", "nn_other", "na_other", "ns_other", "pv_other",
    };
    for (unsigned i = 0; i < sizeof(profiles) / sizeof(profiles[0]); ++i) {
        nvs_handle_t h;
        TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_open(profiles[i], NVS_READWRITE, &h));
        TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_set_str(h, "sentinel", profiles[i]));
        nvs_close(h);
    }
    cfg_wifi_t wifi = {0}, restored_wifi = {0};
    strcpy(wifi.ssid_1, "retained-network"); strcpy(wifi.psw_1, "test-password");
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_set_wifi(&wifi));
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_set_legacy_wifi_primary(&wifi));
    cfg_lcd_t lcd = {.brightness=12, .rotation=1};
    strcpy(lcd.idle_mode, "image");
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_set_lcd(&lcd));
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_factory_reset_preserve());
    for (unsigned i = 0; i < sizeof(profiles) / sizeof(profiles[0]); ++i) {
        nvs_handle_t h; char value[32] = {0}; size_t size=sizeof(value);
        TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_open(profiles[i], NVS_READONLY, &h));
        TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_get_str(h, "sentinel", value, &size));
        TEST_ASSERT_EQUAL_STRING(profiles[i], value); nvs_close(h);
    }
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_get_wifi(&restored_wifi));
    TEST_ASSERT_EQUAL_STRING(wifi.ssid_1, restored_wifi.ssid_1);
    TEST_ASSERT_EQUAL_STRING(wifi.psw_1, restored_wifi.psw_1);
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_get_lcd(&lcd));
    TEST_ASSERT_EQUAL_INT(80, lcd.brightness);
    TEST_ASSERT_EQUAL_STRING("qr", lcd.idle_mode);
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_factory_reset_preserve());
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_factory_reset_all());
    nvs_handle_t h;
    TEST_ASSERT_EQUAL_INT(ESP_ERR_NVS_NOT_FOUND, nvs_open("nyx", NVS_READONLY, &h));
    TEST_ASSERT_EQUAL_INT(ESP_ERR_NVS_NOT_FOUND, nvs_open("wifi", NVS_READONLY, &h));
}

void test_failed_config_reset_reports_error_without_losing_identity(void)
{
    nyx_binding_t node = binding();
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_store_binding(&node));
    cfg_lcd_t lcd = {.brightness=12};
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_config_set_lcd(&lcd));
    fail_erase = true;
    TEST_ASSERT_EQUAL_INT(ESP_FAIL, nvs_config_factory_reset_preserve());
    TEST_ASSERT_TRUE(nyx_creds_read_binding(&node));
    TEST_ASSERT_EQUAL_STRING("nyx_nauth_new", node.auth_token);
}

void test_node_profile_never_reads_or_overwrites_other_server(void)
{
    nvs_handle_t h;
    const char *other = strcmp(NYX_NODE_NVS_NAMESPACE, "nyx") ? "nyx" : "nn_other";
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_open(other, NVS_READWRITE, &h));
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_set_str(h, "api_key", "other-server-token"));
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_set_str(h, "node_id", "other-server-node"));
    TEST_ASSERT_FALSE(nyx_creds_load());
    nyx_binding_t node = binding();
    TEST_ASSERT_EQUAL_INT(0, nyx_creds_store_binding(&node));
    TEST_ASSERT_TRUE(nyx_creds_load());
    TEST_ASSERT_EQUAL_STRING(node.auth_token, nyx_creds_api_key());
    char value[64]={0}; size_t size=sizeof(value);
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_get_str(h, "api_key", value, &size));
    TEST_ASSERT_EQUAL_STRING("other-server-token", value);
    nvs_close(h);
}

void test_corrupt_agent_blob_is_not_reported_as_signed_in(void)
{
    nvs_handle_t h;
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_open(NYX_AGENT_NVS_NAMESPACE, NVS_READWRITE, &h));
    nyx_agent_credential_t corrupt;
    memset(&corrupt, 'x', sizeof(corrupt));
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_set_blob(h, "credential_v1", &corrupt, sizeof(corrupt)));
    TEST_ASSERT_FALSE(nyx_creds_have_agent());
    TEST_ASSERT_FALSE(nyx_creds_read_agent(&corrupt));
    TEST_ASSERT_EQUAL_INT(0, corrupt.credential[0]);
    TEST_ASSERT_EQUAL_INT(ESP_OK, nvs_set_blob(h, "credential_v1", "short", 6));
    TEST_ASSERT_FALSE(nyx_creds_have_agent());
    nvs_close(h);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_identity_activates_on_reload_and_survives_preserving_reset);
    RUN_TEST(test_failed_write_retains_legacy_and_uart_updates_new_format);
    RUN_TEST(test_agent_key_is_independent_atomic_and_preserved_with_identity);
    RUN_TEST(test_preserving_reset_keeps_inactive_profiles_wifi_and_pending_setup);
    RUN_TEST(test_failed_config_reset_reports_error_without_losing_identity);
    RUN_TEST(test_node_profile_never_reads_or_overwrites_other_server);
    RUN_TEST(test_corrupt_agent_blob_is_not_reported_as_signed_in);
    return UNITY_END();
}
