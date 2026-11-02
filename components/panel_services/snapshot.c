#include "snapshot.h"

#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include <string.h>

static struct {
    SemaphoreHandle_t lock;
    uint8_t *buf;
    size_t len;
} g;

void snapshot_init(void)
{
    if (!g.lock) {
        g.lock = xSemaphoreCreateMutex();
    }
    if (!g.buf) {
        g.buf = heap_caps_malloc(SNAPSHOT_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!g.buf) {
            g.buf = heap_caps_malloc(SNAPSHOT_MAX_BYTES, MALLOC_CAP_8BIT);
        }
    }
}

esp_err_t snapshot_store_jpeg(const uint8_t *data, size_t len)
{
    if (!data || len == 0u || len > SNAPSHOT_MAX_BYTES) {
        return ESP_ERR_INVALID_ARG;
    }
    snapshot_init();
    if (!g.buf || !g.lock) {
        return ESP_ERR_NO_MEM;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    memcpy(g.buf, data, len);
    g.len = len;
    xSemaphoreGive(g.lock);
    return ESP_OK;
}

esp_err_t snapshot_copy_latest(uint8_t *dst, size_t cap, size_t *out_len)
{
    if (!dst || !out_len) {
        return ESP_ERR_INVALID_ARG;
    }
    snapshot_init();
    if (!g.buf || !g.lock || g.len == 0u) {
        *out_len = 0u;
        return ESP_ERR_NOT_FOUND;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    if (g.len > cap) {
        xSemaphoreGive(g.lock);
        return ESP_ERR_NO_MEM;
    }
    memcpy(dst, g.buf, g.len);
    *out_len = g.len;
    xSemaphoreGive(g.lock);
    return ESP_OK;
}
