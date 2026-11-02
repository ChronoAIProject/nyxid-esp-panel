#include "media_mode.h"

#include "common.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "heap_diag.h"
#include "stream_publisher.h"

static const char *TAG = "haCam.mode";

static struct {
    SemaphoreHandle_t lock;
    media_mode_t current;
    bool initialized;
} g;

const char *media_mode_name(media_mode_t mode)
{
    switch (mode) {
    case MEDIA_MODE_PREVIEW:
        return "preview";
    case MEDIA_MODE_PUBLISH:
    default:
        return "publish";
    }
}

void media_mode_init(void)
{
    if (!g.lock) {
        g.lock = xSemaphoreCreateMutex();
    }
    g.current = MEDIA_MODE_PUBLISH;
    g.initialized = true;
}

media_mode_t media_mode_current(void)
{
    if (!g.lock) {
        return MEDIA_MODE_PUBLISH;
    }
    media_mode_t mode = MEDIA_MODE_PUBLISH;
    if (xSemaphoreTake(g.lock, pdMS_TO_TICKS(200)) == pdTRUE) {
        mode = g.current;
        xSemaphoreGive(g.lock);
    }
    return mode;
}

bool media_mode_is_preview(void)
{
    return media_mode_current() == MEDIA_MODE_PREVIEW;
}

esp_err_t media_mode_switch(media_mode_t mode, const char *reason)
{
    if (!g.initialized) {
        media_mode_init();
    }
    if (xSemaphoreTake(g.lock, portMAX_DELAY) != pdTRUE) {
        return ESP_ERR_TIMEOUT;
    }

    if (g.current == mode) {
        xSemaphoreGive(g.lock);
        return ESP_OK;
    }

    media_mode_t before = g.current;
    ESP_LOGI(TAG, "switch %s -> %s reason=%s",
             media_mode_name(before), media_mode_name(mode), reason ? reason : "");
    heap_diag_log(TAG, "mode_switch_before");

    esp_err_t err = ESP_OK;
    if (before == MEDIA_MODE_PUBLISH) {
        stream_publisher_suspend_for_preview();
        stop_webrtc();
        media_sys_capture_close();
    } else {
        media_sys_preview_stop();
        media_sys_capture_close();
    }

    vTaskDelay(pdMS_TO_TICKS(500));

    if (mode == MEDIA_MODE_PREVIEW) {
        err = media_sys_prepare_preview_capture();
        if (err == ESP_OK) {
            err = media_sys_preview_start();
        }
    } else {
        err = media_sys_prepare_publish_capture(VIDEO_FPS);
    }

    if (err == ESP_OK) {
        g.current = mode;
        if (mode == MEDIA_MODE_PUBLISH) {
            stream_publisher_resume_after_preview();
        }
        ESP_LOGI(TAG, "mode active=%s dma_largest=%lu",
                 media_mode_name(mode),
                 (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL));
        heap_diag_log(TAG, "mode_switch_after");
    } else {
        ESP_LOGE(TAG, "mode switch to %s failed: %s", media_mode_name(mode), esp_err_to_name(err));
        heap_diag_log(TAG, "mode_switch_failed");
    }

    xSemaphoreGive(g.lock);
    return err;
}

esp_err_t media_mode_request_publish(const char *reason)
{
    return media_mode_switch(MEDIA_MODE_PUBLISH, reason);
}

esp_err_t media_mode_request_preview(const char *reason)
{
    return media_mode_switch(MEDIA_MODE_PREVIEW, reason);
}
