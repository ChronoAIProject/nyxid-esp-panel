#pragma once
#include "esp_codec_dev.h"
#include "driver/i2s_std.h"
#include "esp_err.h"

// Implemented by exactly one source selected by the hardware profile.
int product_board_display_brightness_get(void);
esp_err_t product_board_i2s_read(esp_codec_dev_handle_t record_dev_handle, void *audio_buffer,
                                size_t len, size_t *bytes_read, uint32_t timeout_ms);
esp_err_t product_board_codec_set_fs(esp_codec_dev_handle_t play_dev_handle,
    esp_codec_dev_handle_t record_dev_handle, uint32_t rate, uint32_t bits_cfg, i2s_slot_mode_t ch);
