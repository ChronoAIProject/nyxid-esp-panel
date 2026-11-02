#include "device_code.h"

#include "esp_log.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#ifdef CONFIG_HA_CAM_DEVICE_CODE_MOCK
static const char *TAG = "haCam.devcode.mock";

static struct {
    uint32_t request_count;
    uint32_t poll_count;
} g_mock;

static void random_uuid(char out[37])
{
    uint8_t b[16];
    for (size_t i = 0; i < sizeof(b); i += 4u) {
        uint32_t r = esp_random();
        size_t copy = sizeof(b) - i;
        if (copy > sizeof(r)) {
            copy = sizeof(r);
        }
        memcpy(b + i, &r, copy);
    }
    b[6] = (uint8_t)((b[6] & 0x0fu) | 0x40u);
    b[8] = (uint8_t)((b[8] & 0x3fu) | 0x80u);
    snprintf(out, 37,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

esp_err_t device_code_client_request(const char *device_pubkey_hex,
                                     const char *hw_id,
                                     const char *suggested_label,
                                     device_code_request_result_t *out)
{
    if (!device_pubkey_hex || !hw_id || !suggested_label || !out) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_LOGI(TAG, "mock request pubkey=%.12s hw_id=%s label=%s",
             device_pubkey_hex, hw_id, suggested_label);
    vTaskDelay(pdMS_TO_TICKS(350));
    memset(out, 0, sizeof(*out));
    g_mock.request_count++;
    g_mock.poll_count = 0;
    snprintf(out->device_code, sizeof(out->device_code), "mock-device-code-%08" PRIx32,
             esp_random());
    snprintf(out->user_code, sizeof(out->user_code), "ABCD-EFGH-JKMN");
    snprintf(out->verification_uri, sizeof(out->verification_uri),
             "https://nyx.chrono-ai.fun/devices/onboard");
    out->expires_in = 30u;
    out->poll_interval = 5u;
    return ESP_OK;
}

esp_err_t device_code_client_poll(const char *device_code,
                                  int64_t ts,
                                  const char *ts_sig_hex,
                                  device_code_poll_result_t *out)
{
    if (!device_code || !ts_sig_hex || !out) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_LOGI(TAG, "mock poll code=%s ts=%" PRId64 " sig=%.12s attempt=%lu",
             device_code, ts, ts_sig_hex, (unsigned long)(g_mock.poll_count + 1u));
    vTaskDelay(pdMS_TO_TICKS(200));
    memset(out, 0, sizeof(*out));
    g_mock.poll_count++;
    if (g_mock.poll_count <= 5u) {
        out->status = DEVICE_CODE_POLL_PENDING;
        return ESP_OK;
    }

    out->status = DEVICE_CODE_POLL_APPROVED;
    char uuid[37];
    random_uuid(uuid);
    snprintf(out->api_key, sizeof(out->api_key), "nyx_nauth_MOCK_%08" PRIx32 "%08" PRIx32,
             esp_random(), esp_random());
    snprintf(out->node_id, sizeof(out->node_id), "%s", uuid);
    snprintf(out->refresh_token, sizeof(out->refresh_token), "nyx_refresh_MOCK_%08" PRIx32,
             esp_random());
    return ESP_OK;
}
#endif
