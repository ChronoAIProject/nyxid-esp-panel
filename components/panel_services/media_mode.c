#include "media_mode.h"
#include "panel_ota.h"
#include "camera_capture.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <stdatomic.h>

static SemaphoreHandle_t owner;
static atomic_int current;
static atomic_uint completed, failed, last_capture_ms;
static atomic_int last_error;

const char *media_mode_name(media_mode_t mode)
{
    switch (mode) {
    case MEDIA_MODE_IDLE: return "idle";
    case MEDIA_MODE_PREVIEW: return "preview";
    case MEDIA_MODE_CAPTURE: return "capture";
    default: return "error";
    }
}

void media_mode_init(void)
{
    // Called by service startup before routes and UI become available.
    if (!owner) owner = xSemaphoreCreateMutex();
}

media_mode_t media_mode_current(void) { return atomic_load(&current); }
bool media_mode_is_preview(void) { return media_mode_current() == MEDIA_MODE_PREVIEW; }

void media_mode_status(camera_status_t *status)
{
    *status = (camera_status_t) {
        .mode = media_mode_current(), .completed = atomic_load(&completed),
        .failed = atomic_load(&failed), .last_capture_ms = atomic_load(&last_capture_ms),
        .last_error = atomic_load(&last_error),
    };
}

static esp_err_t close_camera(void)
{
    esp_err_t err = media_sys_capture_close();
    atomic_store(&current, err == ESP_OK ? MEDIA_MODE_IDLE : MEDIA_MODE_ERROR);
    return err;
}

static esp_err_t start_camera(void)
{
    esp_err_t err = media_sys_prepare_preview_capture();
    if (err == ESP_OK) err = media_sys_preview_start();
    if (err != ESP_OK) {
        esp_err_t cleanup = close_camera();
        if (cleanup != ESP_OK) return cleanup;
    }
    return err;
}

esp_err_t media_mode_request_idle(const char *reason)
{
    (void)reason;
    if (!owner) return ESP_ERR_NO_MEM;
    xSemaphoreTake(owner, portMAX_DELAY);
    esp_err_t err = media_mode_current() == MEDIA_MODE_IDLE ? ESP_OK : close_camera();
    atomic_store(&last_error, err);
    xSemaphoreGive(owner);
    return err;
}

esp_err_t media_mode_request_preview(const char *reason)
{
    (void)reason;
    if (!owner) return ESP_ERR_NO_MEM;
    xSemaphoreTake(owner, portMAX_DELAY);
    esp_err_t err = ESP_OK;
    if (!media_mode_is_preview()) {
        media_mode_t previous=media_mode_current();
        atomic_store(&current,MEDIA_MODE_PREVIEW);
        if (panel_ota_busy()) {
            atomic_store(&current,previous);xSemaphoreGive(owner);return ESP_ERR_INVALID_STATE;
        }
        err = start_camera();
        if (err == ESP_OK) atomic_store(&current, MEDIA_MODE_PREVIEW);
    }
    atomic_store(&last_error, err);
    xSemaphoreGive(owner);
    return err;
}

esp_err_t media_mode_snapshot(uint8_t *dst, size_t cap, size_t *out_len, bool allow_start)
{
    if (out_len) *out_len = 0;
    if (!dst || !cap || !out_len) return ESP_ERR_INVALID_ARG;
    if (!owner) return ESP_ERR_INVALID_STATE;
    // No unbounded queue of camera work or response buffers.
    if (xSemaphoreTake(owner, 0) != pdTRUE) return ESP_ERR_TIMEOUT;
    bool preview = media_mode_is_preview();
    if (!preview && !allow_start) {
        xSemaphoreGive(owner);
        return ESP_ERR_INVALID_STATE;
    }
    if (!preview) atomic_store(&current,MEDIA_MODE_CAPTURE);
    if (panel_ota_busy()) {
        if (!preview) atomic_store(&current,MEDIA_MODE_IDLE);
        xSemaphoreGive(owner);return ESP_ERR_INVALID_STATE;
    }
    int64_t started = esp_timer_get_time();
    esp_err_t err = ESP_OK;
    if (!preview) {
        err = start_camera();
    }
    if (err == ESP_OK) err = media_sys_preview_snapshot_encode(dst, cap, out_len);
    if (!preview) {
        esp_err_t cleanup = close_camera();
        if (cleanup != ESP_OK) err = cleanup;
    }
    if (err == ESP_OK) atomic_fetch_add(&completed, 1);
    else { *out_len = 0; atomic_fetch_add(&failed, 1); }
    atomic_store(&last_error, err);
    atomic_store(&last_capture_ms, (esp_timer_get_time() - started) / 1000);
    xSemaphoreGive(owner);
    return err;
}
