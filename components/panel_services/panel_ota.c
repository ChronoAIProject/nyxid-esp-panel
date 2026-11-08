#include "panel_ota.h"
#include <stdatomic.h>
static atomic_bool busy;
static _Atomic(const char *) stage = "idle";
static atomic_int last_error;
bool panel_ota_try_lock(void)
{
    bool expected = false;
    return atomic_compare_exchange_strong(&busy, &expected, true);
}
void panel_ota_unlock(void) { atomic_store(&busy, false); }
bool panel_ota_busy(void) { return atomic_load(&busy); }
void panel_ota_progress(const char *value, esp_err_t error)
{
    atomic_store(&last_error, error);
    atomic_store(&stage, value);
}
const char *panel_ota_stage(void) { return atomic_load(&stage); }
esp_err_t panel_ota_error(void) { return atomic_load(&last_error); }

#include "esp_app_desc.h"
#include "esp_app_format.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "panel_flash.h"
#include "panel_update.h"
#include "panel_voice.h"
#include "media_mode.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mbedtls/sha256.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const char *TAG = "panel.ota";
static bool sha256_matches(const uint8_t a[32], const uint8_t b[32])
{
    unsigned diff=0;
    for (size_t i=0;i<32;++i) diff |= a[i]^b[i];
    return diff==0;
}
typedef struct {
    char *url;
    uint8_t expected[32];
    const esp_partition_t *update;
    char version[32];
    size_t expected_size;
} url_ota_job_t;

static void url_ota_task(void *arg)
{
    url_ota_job_t *job = arg;
    // URLs may contain signed query credentials; log only the target partition.
    ESP_LOGI(TAG, "URL OTA downloading target=%s", job->update->label);
    panel_ota_progress("downloading", ESP_OK);
    esp_http_client_config_t http_cfg = {
        .url = job->url,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .timeout_ms = 15000,
        .keep_alive_enable = true,
        // Use the same bounded flash write path as the raw OTA endpoint.
        .buffer_size = 4096,
    };
    if (job->version[0]) http_cfg.disable_auto_redirect = true;
    esp_http_client_handle_t client = esp_http_client_init(&http_cfg);
    esp_err_t err = client ? ESP_OK : ESP_ERR_NO_MEM;
    int64_t image_len = 0;
    for (unsigned redirects = 0; err == ESP_OK; ++redirects) {
        err = esp_http_client_open(client, 0);
        if (err != ESP_OK) break;
        ESP_LOGI(TAG, "URL OTA HTTP connected");
        image_len = esp_http_client_fetch_headers(client);
        int status = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "URL OTA HTTP status=%d bytes=%lld", status, (long long)image_len);
        if (status == 200) break;
        if (redirects >= 3 || (status != 301 && status != 302 && status != 303 &&
                               status != 307 && status != 308)) {
            err = ESP_ERR_INVALID_RESPONSE;
            break;
        }
        err = esp_http_client_set_redirection(client);
        if (err == ESP_OK && job->version[0]) {
            char redirected[2049];
            err = esp_http_client_get_url(client, redirected, sizeof(redirected));
            if (err == ESP_OK && !panel_update_download_url_allowed(redirected)) err = ESP_ERR_INVALID_ARG;
        }
        esp_http_client_close(client);
    }
    if (err == ESP_OK && (image_len <= 0 || image_len > job->update->size ||
        (job->expected_size && image_len != job->expected_size))) {
        err = ESP_ERR_INVALID_SIZE;
    }
    uint8_t *buffer = heap_caps_malloc(4096, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!buffer) err = ESP_ERR_NO_MEM;
    uint8_t *image = err == ESP_OK ? heap_caps_malloc((size_t)image_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : NULL;
    if (err == ESP_OK && !image) err = ESP_ERR_NO_MEM;
    esp_ota_handle_t handle = 0;
    mbedtls_sha256_context hash;
    mbedtls_sha256_init(&hash);
    if (mbedtls_sha256_starts(&hash, 0) != 0) err = ESP_FAIL;
    size_t received = 0;
    int64_t deadline = esp_timer_get_time() + 300000000LL;
    while (err == ESP_OK && received < (size_t)image_len) {
        size_t wanted = (size_t)image_len - received;
        if (wanted > 4096) wanted = 4096;
        int count = esp_http_client_read(client, (char *)buffer, wanted);
        if (count <= 0 || esp_timer_get_time() > deadline) {
            err = ESP_ERR_TIMEOUT;
            break;
        }
        if (mbedtls_sha256_update(&hash, buffer, count) != 0) {
            err = ESP_FAIL;
            break;
        }
        memcpy(image + received, buffer, count);
        received += count;
        vTaskDelay(1);
    }
    uint8_t actual[32] = {0};
    if (err == ESP_OK) {
        panel_ota_progress("verifying", ESP_OK);
        if (mbedtls_sha256_finish(&hash, actual) != 0) err = ESP_FAIL;
        else if (!sha256_matches(job->expected, actual)) err = ESP_ERR_INVALID_CRC;
    }
    mbedtls_sha256_free(&hash);
    if (client) esp_http_client_cleanup(client);
    if (err == ESP_OK && job->version[0]) {
        const size_t description_offset = sizeof(esp_image_header_t) + sizeof(esp_image_segment_header_t);
        if (received < description_offset + sizeof(esp_app_desc_t)) err = ESP_ERR_INVALID_SIZE;
        else {
            esp_app_desc_t description;
            memcpy(&description, image + description_offset, sizeof(description));
            if (description.magic_word != ESP_APP_DESC_MAGIC_WORD ||
                strncmp(description.project_name, "nyxid-esp-panel", sizeof(description.project_name)) ||
                strncmp(description.version, job->version, sizeof(description.version)))
                err = ESP_ERR_INVALID_VERSION;
        }
    }
    // Close the download transport and verify its complete body before flash I/O.
    // The buffer is bounded by the inactive slot and available PSRAM.
    if (err == ESP_OK) {
        panel_ota_progress("writing", ESP_OK);
        // Bound each erase to the next 4 KiB write, rather than blocking in
        // one multi-megabyte erase on the legacy P4 while the UI is running.
        ESP_LOGI(TAG, "URL OTA download verified; starting sector writes");
        panel_flash_lock();
        err = esp_ota_begin(job->update, OTA_WITH_SEQUENTIAL_WRITES, &handle);
        panel_flash_unlock();
        if (err == ESP_OK) ESP_LOGI(TAG, "URL OTA writer ready");
    }
    for (size_t offset = 0; err == ESP_OK && offset < received;) {
        size_t count = received - offset;
        if (count > 4096) count = 4096;
        memcpy(buffer, image + offset, count);
        err = esp_ota_write(handle, buffer, count);
        offset += count;
        if (err == ESP_OK && ((offset % (1024 * 1024)) == 0 || offset == received)) {
            ESP_LOGI(TAG, "URL OTA written %u/%u bytes", (unsigned)offset, (unsigned)received);
        }
        vTaskDelay(1);
    }
    free(buffer);
    free(image);
    if (err == ESP_OK) {
        panel_flash_lock();
        err = esp_ota_end(handle);
        handle = 0;
        if (err == ESP_OK && job->version[0]) err = panel_update_record_pending(job->version);
        if (err == ESP_OK) err = esp_ota_set_boot_partition(job->update);
        panel_flash_unlock();
    }
    if (handle) esp_ota_abort(handle);
    free(job->url);
    free(job);
    if (err == ESP_OK) {
        panel_ota_progress("rebooting", ESP_OK);
        ESP_LOGI(TAG, "URL OTA verified; rebooting");
        // Keep the update lock until reboot so no request can overwrite this slot.
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_restart();
    } else {
        panel_ota_progress("failed", err);
        panel_ota_unlock();
        ESP_LOGE(TAG, "URL OTA failed error=%s", esp_err_to_name(err));
    }
    vTaskDelete(NULL);
}

esp_err_t panel_ota_start_url(const char *url, const uint8_t expected[32],
                              const char *version, size_t size)
{
    if (!url || !expected || strlen(url)>2048 ||
        (strncmp(url,"https://",8) && strncmp(url,"http://",7)) ||
        (version && (!panel_update_version_valid(version) || !panel_update_download_url_allowed(url))))
        return ESP_ERR_INVALID_ARG;
    const esp_partition_t *update = esp_ota_get_next_update_partition(NULL);
    if (!update || !panel_ota_try_lock()) return ESP_ERR_INVALID_STATE;
    if (version) {
        panel_voice_snapshot_t voice;panel_voice_snapshot(&voice);
        if (voice.busy || voice.active || media_mode_current()!=MEDIA_MODE_IDLE) {
            panel_ota_unlock();return ESP_ERR_INVALID_STATE;
        }
    }
    url_ota_job_t *job = calloc(1,sizeof(*job));
    if (job) {
        job->url=strdup(url);
        job->update=update;
        memcpy(job->expected,expected,32);
        if (version) snprintf(job->version,sizeof(job->version),"%s",version);
        job->expected_size=size;
    }
    panel_ota_progress("queued",ESP_OK);
    if (!job || !job->url || xTaskCreate(url_ota_task,"url_ota",16*1024,job,5,NULL)!=pdPASS) {
        if (job) free(job->url);
        free(job);
        panel_ota_progress("failed",ESP_ERR_NO_MEM);
        panel_ota_unlock();
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
