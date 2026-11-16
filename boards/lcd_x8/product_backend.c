/* SPDX-License-Identifier: Apache-2.0 */
#include "product_board.h"
#include "product_support.h"
#include "bsp/esp-bsp.h"
#include "esp_check.h"
#include "driver/ledc.h"
#include "esp_codec_dev_defaults.h"
static const char *TAG = "product.board";
#include "product_audio_layout.h"
int product_board_display_brightness_get(void)
{
    uint32_t duty = ledc_get_duty(LEDC_LOW_SPEED_MODE, CONFIG_BSP_DISPLAY_BRIGHTNESS_LEDC_CH);
    return (int)((duty * 100 + 511) / 1023);
}

esp_err_t product_board_i2s_read(esp_codec_dev_handle_t record_dev_handle, void *audio_buffer, size_t len, size_t *bytes_read, uint32_t timeout_ms)
{
    // Capture stays 16-bit M/R even when a media player changes output format.
    // Read complete four-slot TDM frames with one reader and bounded scratch space.
    if (bytes_read) *bytes_read = 0;
    if ((!audio_buffer && len) || len % (2 * sizeof(int16_t))) return ESP_ERR_INVALID_ARG;
    int16_t tdm[128 * 4];
    size_t done = 0;
    while (done < len) {
        size_t frames = (len - done) / (2 * sizeof(int16_t));
        if (frames > 128) frames = 128;
        esp_err_t ret = esp_codec_dev_read(record_dev_handle, tdm, frames * 4 * sizeof(int16_t));
        if (ret != ESP_OK) return ret;
        product_lcd_x8_normalize_capture((int16_t *)((uint8_t *)audio_buffer + done), tdm, frames);
        done += frames * 2 * sizeof(int16_t);
        if (bytes_read) *bytes_read = done;
    }
    return ESP_OK;
}

esp_err_t product_board_codec_set_fs(esp_codec_dev_handle_t play_dev_handle, esp_codec_dev_handle_t record_dev_handle, uint32_t rate, uint32_t bits_cfg, i2s_slot_mode_t ch)
{
    // ES7210 needs all four 16-bit slots for MIC3 to remain the echo reference.
    // Playback shares the clock but retains the format requested by the player.
    ESP_RETURN_ON_FALSE(play_dev_handle && record_dev_handle, ESP_ERR_INVALID_STATE, TAG, "Codec is not initialized");
    ESP_RETURN_ON_FALSE((bits_cfg == 16 || bits_cfg == 32) &&
                       (ch == I2S_SLOT_MODE_MONO || ch == I2S_SLOT_MODE_STEREO),
                       ESP_ERR_NOT_SUPPORTED, TAG, "Unsupported LCD-X8 audio format");
    esp_codec_dev_sample_info_t input_fs = {
        .sample_rate = rate, .channel = 4, .channel_mask = 0xF, .bits_per_sample = 16,
    };
    esp_codec_dev_sample_info_t output_fs = {
        .sample_rate = rate, .channel = ch, .bits_per_sample = bits_cfg,
    };
    esp_err_t ret = esp_codec_dev_close(record_dev_handle);
    if (ret == ESP_OK) ret = esp_codec_dev_close(play_dev_handle);
    if (ret == ESP_OK) ret = esp_codec_dev_open(play_dev_handle, &output_fs);
    if (ret == ESP_OK) ret = esp_codec_dev_open(record_dev_handle, &input_fs);
    if (ret == ESP_OK) ret = esp_codec_dev_set_in_channel_gain(record_dev_handle,
        ES7210_SEL_MIC1 | ES7210_SEL_MIC2, PRODUCT_CODEC_DEFAULT_ADC_VOLUME);
    // MIC3 carries the line-level DAC output; microphone gain would clip it.
    if (ret == ESP_OK) ret = esp_codec_dev_set_in_channel_gain(record_dev_handle, ES7210_SEL_MIC3, 0.0f);
    if (ret != ESP_OK) {
        (void)esp_codec_dev_close(record_dev_handle);
        (void)esp_codec_dev_close(play_dev_handle);
    }
    return ret;
}
