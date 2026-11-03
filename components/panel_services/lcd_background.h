#pragma once

#include "esp_err.h"
#include <stddef.h>
#include <stdint.h>

#define LCD_BACKGROUND_MAX_BYTES 65536u

// One NVS blob: replacement is power-loss safe and independent of app/SPIFFS updates.
// Callers serialize updates and validate the JPEG before writing.
esp_err_t lcd_background_read(uint8_t **jpeg, size_t *len);
esp_err_t lcd_background_write(const uint8_t *jpeg, size_t len);
esp_err_t lcd_background_reset(void);
