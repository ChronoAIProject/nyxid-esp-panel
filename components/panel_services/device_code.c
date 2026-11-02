#include "device_code.h"

#include "cjson_tiny.h"
#include "esp_crt_bundle.h"
#include "esp_check.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "mbedtls/sha512.h"
#include "nyx_creds.h"
#include "settings.h"
#include "nvs.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CONFIG_HA_CAM_DEVICE_CODE
static const char *TAG = "haCam.devcode";
static const char *NYX_NVS_NS = "nyx";
static const uint32_t DEFAULT_EXPIRES_IN_SEC = 900u;
static const uint32_t DEFAULT_POLL_INTERVAL_SEC = 5u;

static struct {
    uint8_t pub[32];
    uint8_t priv[64];
    bool loaded;
    bool in_progress;
    bool task_started;
    bool approved;
    char device_code[96];
    char user_code[16];
    char verification_uri[192];
    char api_key[128];
    char node_id[96];
    uint32_t expires_in;
    uint32_t poll_interval;
    uint32_t poll_attempts;
    int64_t request_us;
    int64_t last_poll_us;
} g;

#ifndef CONFIG_HA_CAM_DEVICE_CODE_MOCK
typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} response_buf_t;

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    response_buf_t *resp = (response_buf_t *)evt->user_data;
    if (evt->event_id != HTTP_EVENT_ON_DATA || !resp || !evt->data || evt->data_len <= 0) {
        return ESP_OK;
    }

    size_t available = (resp->cap > resp->len) ? (resp->cap - resp->len - 1u) : 0u;
    size_t copy_len = (size_t)evt->data_len;
    if (copy_len > available) {
        copy_len = available;
    }
    if (copy_len > 0u) {
        memcpy(resp->buf + resp->len, evt->data, copy_len);
        resp->len += copy_len;
        resp->buf[resp->len] = '\0';
    }
    return ESP_OK;
}

static int json_copy(cJSON *root, const char *key, char *dst, size_t cap)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsString(item) || !item->valuestring || !dst || cap == 0u) {
        return -1;
    }
    snprintf(dst, cap, "%s", item->valuestring);
    return 0;
}

static uint32_t json_u32(cJSON *root, const char *key, uint32_t fallback)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsNumber(item) || item->valuedouble < 0.0) {
        return fallback;
    }
    return (uint32_t)item->valuedouble;
}

static esp_err_t http_post_json(const char *path, const char *body, char *resp_storage,
                                size_t resp_cap, int *status_out)
{
    if (!path || !body || !resp_storage || resp_cap == 0u) {
        return ESP_ERR_INVALID_ARG;
    }
    resp_storage[0] = '\0';

    char url[256];
    snprintf(url, sizeof(url), "%s/api/v1/devices/code/%s",
             CONFIG_HA_CAM_DEVICE_CODE_NYX_BASE, path);

    response_buf_t resp = {
        .buf = resp_storage,
        .len = 0,
        .cap = resp_cap,
    };
    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 10000,
        .event_handler = http_event_handler,
        .user_data = &resp,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (!client) {
        return ESP_ERR_NO_MEM;
    }

    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, body, strlen(body));

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK && status_out) {
        *status_out = esp_http_client_get_status_code(client);
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "device-code HTTP POST %s failed: %s", url, esp_err_to_name(err));
    }
    esp_http_client_cleanup(client);
    return err;
}

static esp_err_t parse_request_response(const char *json, device_code_request_result_t *out)
{
    if (!json || !out) {
        return ESP_ERR_INVALID_ARG;
    }
    cJSON *root = cJSON_Parse(json);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return ESP_ERR_INVALID_RESPONSE;
    }

    esp_err_t err = ESP_OK;
    if (json_copy(root, "device_code", out->device_code, sizeof(out->device_code)) != 0 ||
        json_copy(root, "user_code", out->user_code, sizeof(out->user_code)) != 0 ||
        json_copy(root, "verification_uri", out->verification_uri, sizeof(out->verification_uri)) != 0) {
        err = ESP_ERR_INVALID_RESPONSE;
    }
    out->expires_in = json_u32(root, "expires_in", DEFAULT_EXPIRES_IN_SEC);
    out->poll_interval = json_u32(root, "poll_interval", DEFAULT_POLL_INTERVAL_SEC);
    if (out->poll_interval == 0u) {
        out->poll_interval = DEFAULT_POLL_INTERVAL_SEC;
    }
    cJSON_Delete(root);
    return err;
}

static esp_err_t parse_poll_response(const char *json, device_code_poll_result_t *out)
{
    if (!json || !out) {
        return ESP_ERR_INVALID_ARG;
    }
    cJSON *root = cJSON_Parse(json);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return ESP_ERR_INVALID_RESPONSE;
    }

    out->status = DEVICE_CODE_POLL_PENDING;
    cJSON *status = cJSON_GetObjectItemCaseSensitive(root, "status");
    if (cJSON_IsString(status) && status->valuestring) {
        if (strcmp(status->valuestring, "approved") == 0 ||
            strcmp(status->valuestring, "success") == 0) {
            out->status = DEVICE_CODE_POLL_APPROVED;
        } else if (strcmp(status->valuestring, "expired") == 0) {
            out->status = DEVICE_CODE_POLL_EXPIRED;
        }
    }

    esp_err_t err = ESP_OK;
    if (out->status == DEVICE_CODE_POLL_APPROVED) {
        if (json_copy(root, "api_key", out->api_key, sizeof(out->api_key)) != 0 ||
            json_copy(root, "node_id", out->node_id, sizeof(out->node_id)) != 0) {
            err = ESP_ERR_INVALID_RESPONSE;
        }
        (void)json_copy(root, "refresh_token", out->refresh_token, sizeof(out->refresh_token));
    }
    cJSON_Delete(root);
    return err;
}
#endif

static void bytes_to_hex(const uint8_t *bytes, size_t len, char *out, size_t cap)
{
    static const char chars[] = "0123456789abcdef";
    if (!out || cap == 0u) {
        return;
    }
    if (!bytes || cap < len * 2u + 1u) {
        out[0] = '\0';
        return;
    }
    for (size_t i = 0; i < len; ++i) {
        out[i * 2u] = chars[(bytes[i] >> 4) & 0x0f];
        out[i * 2u + 1u] = chars[bytes[i] & 0x0f];
    }
    out[len * 2u] = '\0';
}

static void copy_string(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0u) {
        return;
    }
    snprintf(dst, cap, "%s", src ? src : "");
}

static void random_bytes(uint8_t *dst, size_t len)
{
    size_t off = 0u;
    while (off < len) {
        uint32_t r = esp_random();
        size_t copy = len - off;
        if (copy > sizeof(r)) {
            copy = sizeof(r);
        }
        memcpy(dst + off, &r, copy);
        off += copy;
    }
}

static void derive_public_key(void)
{
    uint8_t digest[64];
    (void)mbedtls_sha512(g.priv, 32, digest, 0);
    memcpy(g.pub, digest, sizeof(g.pub));
}

static esp_err_t load_keypair_from_nvs(nvs_handle_t h)
{
    size_t pub_len = sizeof(g.pub);
    size_t priv_len = sizeof(g.priv);
    esp_err_t err = nvs_get_blob(h, "dc_pub", g.pub, &pub_len);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_get_blob(h, "dc_priv", g.priv, &priv_len);
    if (err != ESP_OK) {
        return err;
    }
    if (pub_len != sizeof(g.pub) || priv_len != sizeof(g.priv)) {
        return ESP_ERR_INVALID_SIZE;
    }
    return ESP_OK;
}

static esp_err_t generate_and_save_keypair(nvs_handle_t h)
{
    random_bytes(g.priv, sizeof(g.priv));
    derive_public_key();

    esp_err_t err = nvs_set_blob(h, "dc_pub", g.pub, sizeof(g.pub));
    if (err == ESP_OK) {
        err = nvs_set_blob(h, "dc_priv", g.priv, sizeof(g.priv));
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    if (err != ESP_OK) {
        return err;
    }
    ESP_LOGI(TAG, "generated device-code keypair namespace=%s keys=dc_pub,dc_priv", NYX_NVS_NS);
    return ESP_OK;
}

static esp_err_t ensure_keypair(void)
{
    if (g.loaded) {
        return ESP_OK;
    }

    nvs_handle_t h = 0;
    ESP_RETURN_ON_ERROR(nvs_open(NYX_NVS_NS, NVS_READWRITE, &h), TAG, "nvs_open(%s)", NYX_NVS_NS);

    esp_err_t err = load_keypair_from_nvs(h);
    if (err == ESP_ERR_NVS_NOT_FOUND || err == ESP_ERR_INVALID_SIZE) {
        ESP_LOGI(TAG, "device-code keypair missing or invalid; generating");
        err = generate_and_save_keypair(h);
    }
    nvs_close(h);
    ESP_RETURN_ON_ERROR(err, TAG, "device-code keypair load/generate failed");

    g.loaded = true;
    return ESP_OK;
}

static void device_code_poll_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "device-code polling task started");

    esp_err_t err = device_code_request();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "device-code request failed: %s", esp_err_to_name(err));
        g.in_progress = false;
        vTaskDelete(NULL);
        return;
    }

    while (g.in_progress && !g.approved) {
        uint32_t interval = g.poll_interval ? g.poll_interval : DEFAULT_POLL_INTERVAL_SEC;
        vTaskDelay(pdMS_TO_TICKS(interval * 1000u));
        err = device_code_poll();
        if (err == ESP_ERR_NOT_FINISHED) {
            continue;
        }
        if (err == ESP_ERR_TIMEOUT) {
            ESP_LOGW(TAG, "device-code expired; rotating request");
            err = device_code_request();
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "device-code rotate request failed: %s", esp_err_to_name(err));
                g.in_progress = false;
                break;
            }
            continue;
        }
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "device-code poll failed: %s", esp_err_to_name(err));
            continue;
        }

        if (nyx_creds_save_api_key(g.api_key) != 0 ||
            nyx_creds_save_node_id(g.node_id) != 0) {
            ESP_LOGE(TAG, "device-code credentials save failed");
            g.in_progress = false;
            break;
        }
        ESP_LOGI(TAG, "device-code approved; saved credentials and rebooting");
        vTaskDelay(pdMS_TO_TICKS(500));
        esp_restart();
    }

    g.task_started = false;
    vTaskDelete(NULL);
}

esp_err_t device_code_start_if_needed(void)
{
    ESP_RETURN_ON_ERROR(ensure_keypair(), TAG, "keypair unavailable");
    if (g.task_started) {
        g.in_progress = true;
        return ESP_OK;
    }
    g.in_progress = true;
    g.approved = false;
    BaseType_t ok = xTaskCreate(device_code_poll_task, "devcode_poll", 8192, NULL, 5, NULL);
    if (ok != pdPASS) {
        g.in_progress = false;
        return ESP_ERR_NO_MEM;
    }
    g.task_started = true;
    return ESP_OK;
}

void device_code_cancel(void)
{
    g.in_progress = false;
    ESP_LOGI(TAG, "device-code flow canceled");
}

bool device_code_in_progress(void)
{
    return g.in_progress;
}

void device_code_get_user_code(char out[16])
{
    if (out) {
        snprintf(out, 16, "%s", g.user_code);
    }
}

void device_code_get_verification_uri(char *out, size_t cap)
{
    if (out && cap > 0u) {
        snprintf(out, cap, "%s", g.verification_uri);
    }
}

uint32_t device_code_seconds_until_rotate(void)
{
    if (!g.in_progress || g.request_us == 0 || g.expires_in == 0u) {
        return 0u;
    }
    uint64_t elapsed = (uint64_t)((esp_timer_get_time() - g.request_us) / 1000000LL);
    if (elapsed >= g.expires_in) {
        return 0u;
    }
    return g.expires_in - (uint32_t)elapsed;
}

uint32_t device_code_last_poll_age_seconds(void)
{
    if (g.last_poll_us == 0) {
        return 0u;
    }
    int64_t age = (esp_timer_get_time() - g.last_poll_us) / 1000000LL;
    return age > 0 ? (uint32_t)age : 0u;
}

uint32_t device_code_poll_attempts(void)
{
    return g.poll_attempts;
}

esp_err_t device_code_sign(const uint8_t *msg, size_t len, uint8_t out[64])
{
    if (!msg || !out) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_RETURN_ON_ERROR(ensure_keypair(), TAG, "keypair unavailable");

    mbedtls_sha512_context ctx;
    mbedtls_sha512_init(&ctx);
    int rc = mbedtls_sha512_starts(&ctx, 0);
    if (rc == 0) {
        rc = mbedtls_sha512_update(&ctx, g.priv, sizeof(g.priv));
    }
    if (rc == 0) {
        rc = mbedtls_sha512_update(&ctx, msg, len);
    }
    if (rc == 0) {
        rc = mbedtls_sha512_finish(&ctx, out);
    }
    mbedtls_sha512_free(&ctx);
    return rc == 0 ? ESP_OK : ESP_FAIL;
}

esp_err_t device_code_get_public_key_hex(char out[65])
{
    if (!out) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_RETURN_ON_ERROR(ensure_keypair(), TAG, "keypair unavailable");
    bytes_to_hex(g.pub, sizeof(g.pub), out, 65);
    return ESP_OK;
}

esp_err_t device_code_request(void)
{
    device_code_request_result_t result = {0};
    char pub_hex[65];
    ESP_RETURN_ON_ERROR(device_code_get_public_key_hex(pub_hex), TAG, "pubkey unavailable");

    uint8_t mac[6] = {0};
    if (esp_read_mac(mac, ESP_MAC_WIFI_STA) != ESP_OK) {
        random_bytes(mac, sizeof(mac));
    }
    char hw_id[40];
    snprintf(hw_id, sizeof(hw_id), "esp32p4-%02x%02x%02x%02x%02x%02x",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    esp_err_t err = device_code_client_request(pub_hex, hw_id, HA_CAM_DEFAULT_LABEL, &result);
    if (err != ESP_OK) {
        return err;
    }

    copy_string(g.device_code, sizeof(g.device_code), result.device_code);
    copy_string(g.user_code, sizeof(g.user_code), result.user_code);
    copy_string(g.verification_uri, sizeof(g.verification_uri), result.verification_uri);
    g.expires_in = result.expires_in ? result.expires_in : DEFAULT_EXPIRES_IN_SEC;
    g.poll_interval = result.poll_interval ? result.poll_interval : DEFAULT_POLL_INTERVAL_SEC;
    g.poll_attempts = 0u;
    g.last_poll_us = 0;
    g.request_us = esp_timer_get_time();
    ESP_LOGI(TAG, "device-code requested user_code=%s verification_uri=%s poll_interval=%lu expires_in=%lu",
             g.user_code, g.verification_uri, (unsigned long)g.poll_interval,
             (unsigned long)g.expires_in);
    return ESP_OK;
}

esp_err_t device_code_poll(void)
{
    if (!g.device_code[0]) {
        return ESP_ERR_INVALID_STATE;
    }

    char msg[160];
    int64_t ts = esp_timer_get_time() / 1000000LL;
    snprintf(msg, sizeof(msg), "%s:%" PRId64, g.device_code, ts);
    uint8_t sig[64];
    ESP_RETURN_ON_ERROR(device_code_sign((const uint8_t *)msg, strlen(msg), sig), TAG, "sign failed");
    char sig_hex[129];
    bytes_to_hex(sig, sizeof(sig), sig_hex, sizeof(sig_hex));

    device_code_poll_result_t result = {0};
    g.poll_attempts++;
    g.last_poll_us = esp_timer_get_time();
    esp_err_t err = device_code_client_poll(g.device_code, ts, sig_hex, &result);
    if (err != ESP_OK) {
        return err;
    }
    if (result.status == DEVICE_CODE_POLL_PENDING) {
        return ESP_ERR_NOT_FINISHED;
    }
    if (result.status == DEVICE_CODE_POLL_EXPIRED) {
        return ESP_ERR_TIMEOUT;
    }

    copy_string(g.api_key, sizeof(g.api_key), result.api_key);
    copy_string(g.node_id, sizeof(g.node_id), result.node_id);
    g.approved = g.api_key[0] && g.node_id[0];
    return g.approved ? ESP_OK : ESP_ERR_INVALID_RESPONSE;
}

esp_err_t device_code_get_credentials(char *api_key_out, char *node_id_out, size_t cap)
{
    if (!api_key_out || !node_id_out || cap == 0u) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!g.approved || !g.api_key[0] || !g.node_id[0]) {
        return ESP_ERR_INVALID_STATE;
    }
    copy_string(api_key_out, cap, g.api_key);
    copy_string(node_id_out, cap, g.node_id);
    return ESP_OK;
}

#ifndef CONFIG_HA_CAM_DEVICE_CODE_MOCK
esp_err_t device_code_client_request(const char *device_pubkey_hex,
                                     const char *hw_id,
                                     const char *suggested_label,
                                     device_code_request_result_t *out)
{
    if (!device_pubkey_hex || !hw_id || !suggested_label || !out) {
        return ESP_ERR_INVALID_ARG;
    }

    cJSON *body = cJSON_CreateObject();
    if (!body) {
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddStringToObject(body, "device_pubkey", device_pubkey_hex);
    cJSON_AddStringToObject(body, "hw_id", hw_id);
    cJSON_AddStringToObject(body, "suggested_label", suggested_label);
    char *printed = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!printed) {
        return ESP_ERR_NO_MEM;
    }

    char resp[768];
    int status = 0;
    esp_err_t err = http_post_json("request", printed, resp, sizeof(resp), &status);
    free(printed);
    if (err != ESP_OK) {
        return err;
    }
    if (status < 200 || status >= 300) {
        ESP_LOGW(TAG, "device-code request HTTP status=%d body=%.200s", status, resp);
        return ESP_FAIL;
    }
    return parse_request_response(resp, out);
}

esp_err_t device_code_client_poll(const char *device_code,
                                  int64_t ts,
                                  const char *ts_sig_hex,
                                  device_code_poll_result_t *out)
{
    if (!device_code || !ts_sig_hex || !out) {
        return ESP_ERR_INVALID_ARG;
    }

    cJSON *body = cJSON_CreateObject();
    if (!body) {
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddStringToObject(body, "device_code", device_code);
    cJSON_AddNumberToObject(body, "ts", (double)ts);
    cJSON_AddStringToObject(body, "ts_sig", ts_sig_hex);
    char *printed = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!printed) {
        return ESP_ERR_NO_MEM;
    }

    char resp[768];
    int status = 0;
    esp_err_t err = http_post_json("poll", printed, resp, sizeof(resp), &status);
    free(printed);
    if (err != ESP_OK) {
        return err;
    }
    if (status == 202 || status == 204) {
        out->status = DEVICE_CODE_POLL_PENDING;
        return ESP_OK;
    }
    if (status < 200 || status >= 300) {
        ESP_LOGW(TAG, "device-code poll HTTP status=%d body=%.200s", status, resp);
        return ESP_FAIL;
    }
    return parse_poll_response(resp, out);
}
#endif
#endif
