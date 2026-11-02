#include "lcd_image.h"

#include "esp_heap_caps.h"
#include "esp_jpeg_dec.h"
#include "esp_log.h"
#include "esp_timer.h"

#include <string.h>

static const char *TAG = "haCam.lcd_image";

#define LCD_IMAGE_MAX_WIDTH 800u
#define LCD_IMAGE_MAX_HEIGHT 1280u

static void log_decode_heap(const char *phase)
{
    ESP_LOGI(TAG, "lcd_image: %s heap largest_free internal=%lu dma=%lu spiram=%lu",
             phase,
             (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
             (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_DMA),
             (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
}

static esp_err_t jpeg_decode_prepare(const uint8_t *jpeg, size_t jpeg_len,
                                     jpeg_dec_handle_t *out_dec, jpeg_dec_io_t *out_io,
                                     jpeg_dec_header_info_t *out_info, int *out_len)
{
    if (!jpeg || jpeg_len == 0u || !out_dec || !out_io || !out_info || !out_len) {
        return ESP_ERR_INVALID_ARG;
    }

    jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
    cfg.output_type = JPEG_PIXEL_FORMAT_RGB565_LE;

    jpeg_dec_handle_t dec = NULL;
    if (jpeg_dec_open(&cfg, &dec) != JPEG_ERR_OK || !dec) {
        ESP_LOGE(TAG, "jpeg_dec_open failed");
        return ESP_FAIL;
    }

    jpeg_dec_io_t io = {
        .inbuf = (uint8_t *)jpeg,
        .inbuf_len = (int)jpeg_len,
    };
    jpeg_dec_header_info_t info = {0};
    jpeg_error_t jerr = jpeg_dec_parse_header(dec, &io, &info);
    if (jerr != JPEG_ERR_OK || info.width == 0 || info.height == 0) {
        ESP_LOGE(TAG, "jpeg header parse failed: %d", jerr);
        jpeg_dec_close(dec);
        return ESP_ERR_INVALID_ARG;
    }
    if (info.width > LCD_IMAGE_MAX_WIDTH || info.height > LCD_IMAGE_MAX_HEIGHT) {
        ESP_LOGE(TAG, "lcd_image: dimensions out of bounds: %ux%u (max %ux%u)",
                 info.width, info.height, LCD_IMAGE_MAX_WIDTH, LCD_IMAGE_MAX_HEIGHT);
        jpeg_dec_close(dec);
        return ESP_ERR_INVALID_SIZE;
    }

    int len = 0;
    jerr = jpeg_dec_get_outbuf_len(dec, &len);
    if (jerr != JPEG_ERR_OK || len <= 0) {
        ESP_LOGE(TAG, "jpeg output size failed: %d", jerr);
        jpeg_dec_close(dec);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "lcd_image: header parsed w=%u h=%u out_len=%u",
             info.width, info.height, (unsigned)len);

    *out_dec = dec;
    *out_io = io;
    *out_info = info;
    *out_len = len;
    return ESP_OK;
}

esp_err_t lcd_image_decode_jpeg(const uint8_t *jpeg, size_t jpeg_len, lcd_image_t *out)
{
    if (!jpeg || jpeg_len == 0u || !out) {
        return ESP_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));

    jpeg_dec_handle_t dec = NULL;
    jpeg_dec_io_t io = {0};
    jpeg_dec_header_info_t info = {0};
    int out_len = 0;
    esp_err_t err = jpeg_decode_prepare(jpeg, jpeg_len, &dec, &io, &info, &out_len);
    if (err != ESP_OK) return err;

    uint8_t *pixels = (uint8_t *)heap_caps_aligned_alloc(16, (size_t)out_len,
                                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    const char *alloc_cap = pixels ? "SPIRAM" : "8BIT";
    if (!pixels) {
        pixels = (uint8_t *)heap_caps_aligned_alloc(16, (size_t)out_len, MALLOC_CAP_8BIT);
    }
    if (!pixels) {
        ESP_LOGI(TAG, "lcd_image: alloc output buf len=%u cap=FAILED spiram_free=%lu internal_free=%lu",
                 (unsigned)out_len,
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        jpeg_dec_close(dec);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "lcd_image: alloc output buf len=%u cap=%s spiram_free=%lu internal_free=%lu",
             (unsigned)out_len, alloc_cap,
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    io.outbuf = pixels;
    io.out_size = out_len;
    log_decode_heap("before decode");
    int64_t decode_start_us = esp_timer_get_time();
    ESP_LOGI(TAG, "lcd_image: decode begin");
    jpeg_error_t jerr = jpeg_dec_process(dec, &io);
    uint32_t elapsed_ms = (uint32_t)((esp_timer_get_time() - decode_start_us) / 1000LL);
    ESP_LOGI(TAG, "lcd_image: decode end rc=%d output_len=%d elapsed_ms=%lu",
             jerr, out_len, (unsigned long)elapsed_ms);
    log_decode_heap("after decode");
    jpeg_dec_close(dec);
    if (jerr != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg decode failed: %d", jerr);
        heap_caps_free(pixels);
        return ESP_ERR_INVALID_RESPONSE;
    }

    out->width = info.width;
    out->height = info.height;
    out->dsc.header.magic = LV_IMAGE_HEADER_MAGIC;
    out->dsc.header.cf = LV_COLOR_FORMAT_RGB565;
    out->dsc.header.flags = 0;
    out->dsc.header.w = info.width;
    out->dsc.header.h = info.height;
    out->dsc.header.stride = (uint16_t)(info.width * 2u);
    out->dsc.data_size = (uint32_t)out_len;
    out->dsc.data = pixels;
    return ESP_OK;
}

esp_err_t lcd_image_decode_jpeg_into(const uint8_t *jpeg, size_t jpeg_len, uint8_t *pixels,
                                     size_t pixel_cap, uint16_t *out_w, uint16_t *out_h,
                                     size_t *out_len)
{
    if (!pixels || pixel_cap == 0u) {
        return ESP_ERR_INVALID_ARG;
    }
    jpeg_dec_handle_t dec = NULL;
    jpeg_dec_io_t io = {0};
    jpeg_dec_header_info_t info = {0};
    int len = 0;
    esp_err_t err = jpeg_decode_prepare(jpeg, jpeg_len, &dec, &io, &info, &len);
    if (err != ESP_OK) return err;
    if ((size_t)len > pixel_cap) {
        jpeg_dec_close(dec);
        return ESP_ERR_NO_MEM;
    }

    io.outbuf = pixels;
    io.out_size = len;
    log_decode_heap("before decode_into");
    int64_t decode_start_us = esp_timer_get_time();
    ESP_LOGI(TAG, "lcd_image: decode begin");
    jpeg_error_t jerr = jpeg_dec_process(dec, &io);
    uint32_t elapsed_ms = (uint32_t)((esp_timer_get_time() - decode_start_us) / 1000LL);
    ESP_LOGI(TAG, "lcd_image: decode end rc=%d output_len=%d elapsed_ms=%lu",
             jerr, len, (unsigned long)elapsed_ms);
    log_decode_heap("after decode_into");
    jpeg_dec_close(dec);
    if (jerr != JPEG_ERR_OK) {
        ESP_LOGE(TAG, "jpeg decode failed: %d", jerr);
        return ESP_ERR_INVALID_RESPONSE;
    }

    if (out_w) *out_w = info.width;
    if (out_h) *out_h = info.height;
    if (out_len) *out_len = (size_t)len;
    return ESP_OK;
}

void lcd_image_free(lcd_image_t *image)
{
    if (!image) {
        return;
    }
    if (image->dsc.data) {
        heap_caps_free((void *)image->dsc.data);
    }
    memset(image, 0, sizeof(*image));
}
