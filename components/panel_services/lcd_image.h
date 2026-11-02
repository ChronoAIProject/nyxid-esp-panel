#pragma once

#include "esp_err.h"
#include "lvgl.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    lv_image_dsc_t dsc;
    uint16_t width;
    uint16_t height;
} lcd_image_t;

esp_err_t lcd_image_decode_jpeg(const uint8_t *jpeg, size_t jpeg_len, lcd_image_t *out);
esp_err_t lcd_image_decode_jpeg_into(const uint8_t *jpeg, size_t jpeg_len, uint8_t *pixels,
                                     size_t pixel_cap, uint16_t *out_w, uint16_t *out_h,
                                     size_t *out_len);
void lcd_image_free(lcd_image_t *image);
