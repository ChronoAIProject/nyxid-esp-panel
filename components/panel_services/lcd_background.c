#include "lcd_background.h"
#include "esp_heap_caps.h"
#include "nvs.h"
#include <stdlib.h>

esp_err_t lcd_background_read(uint8_t **jpeg, size_t *len)
{
    if (!jpeg || !len) return ESP_ERR_INVALID_ARG;
    *jpeg = NULL;
    *len = 0;
    nvs_handle_t h = 0;
    // NVS owns its mutex; individual flash accesses use panel_flash wrappers.
    // Taking panel_flash before NVS would invert Settings' NVS -> flash order.
    esp_err_t err = nvs_open("panel_art", NVS_READONLY, &h);
    size_t bytes = 0;
    if (err == ESP_OK) err = nvs_get_blob(h, "jpeg", NULL, &bytes);
    if (err == ESP_OK && (!bytes || bytes > LCD_BACKGROUND_MAX_BYTES)) err = ESP_ERR_INVALID_SIZE;
    uint8_t *buffer = err == ESP_OK ? heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : NULL;
    if (err == ESP_OK && !buffer) err = ESP_ERR_NO_MEM;
    if (err == ESP_OK) err = nvs_get_blob(h, "jpeg", buffer, &bytes);
    if (h) nvs_close(h);
    if (err != ESP_OK) { free(buffer); return err; }
    *jpeg = buffer;
    *len = bytes;
    return ESP_OK;
}

esp_err_t lcd_background_write(const uint8_t *jpeg, size_t len)
{
    if (!jpeg || !len || len > LCD_BACKGROUND_MAX_BYTES) return ESP_ERR_INVALID_SIZE;
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("panel_art", NVS_READWRITE, &h);
    if (err == ESP_OK) err = nvs_set_blob(h, "jpeg", jpeg, len);
    if (err == ESP_OK) err = nvs_commit(h);
    if (h) nvs_close(h);
    return err;
}

esp_err_t lcd_background_reset(void)
{
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("panel_art", NVS_READWRITE, &h);
    if (err == ESP_OK) {
        err = nvs_erase_key(h, "jpeg");
        if (err == ESP_ERR_NVS_NOT_FOUND) err = ESP_OK;
        if (err == ESP_OK) err = nvs_commit(h);
        nvs_close(h);
    }
    return err;
}
