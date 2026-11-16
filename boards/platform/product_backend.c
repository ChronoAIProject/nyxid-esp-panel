/* SPDX-License-Identifier: Apache-2.0 */
#include "product_board.h"
#include "product_support.h"
#include "bsp/esp-bsp.h"
#include "esp_check.h"
#include "driver/i2c_master.h"
#include "esp_codec_dev_defaults.h"
static const char *TAG = "product.board";
#define PRODUCT_BRIGHTNESS_I2C_ADDR 0x45
#define PRODUCT_BRIGHTNESS_REG 0x96
#define PRODUCT_BRIGHTNESS_DEFAULT 100
int product_board_display_brightness_get(void)
{
    uint8_t reg_addr = PRODUCT_BRIGHTNESS_REG;
    uint8_t data = 0;
    i2c_master_dev_handle_t dev_handle = NULL;
    i2c_master_bus_handle_t bus_handle = NULL;
    i2c_device_config_t i2c_dev_conf = {
        .scl_speed_hz = 100 * 1000,
        .device_address = PRODUCT_BRIGHTNESS_I2C_ADDR,
    };

    esp_err_t ret = bsp_i2c_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "brightness get: init i2c failed, %s", esp_err_to_name(ret));
        return PRODUCT_BRIGHTNESS_DEFAULT;
    }

    bus_handle = bsp_i2c_get_handle();
    if (bus_handle == NULL) {
        ESP_LOGW(TAG, "brightness get: i2c bus handle is NULL");
        return PRODUCT_BRIGHTNESS_DEFAULT;
    }

    ret = i2c_master_bus_add_device(bus_handle, &i2c_dev_conf, &dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "brightness get: add i2c device failed, %s", esp_err_to_name(ret));
        return PRODUCT_BRIGHTNESS_DEFAULT;
    }

    ret = i2c_master_transmit_receive(dev_handle, &reg_addr, sizeof(reg_addr), &data, sizeof(data), 50);
    i2c_master_bus_rm_device(dev_handle);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "brightness get: read i2c register failed, %s", esp_err_to_name(ret));
        return PRODUCT_BRIGHTNESS_DEFAULT;
    }

    return (data * 100 + 127) / 255;
}

esp_err_t product_board_i2s_read(esp_codec_dev_handle_t record_dev_handle, void *audio_buffer, size_t len, size_t *bytes_read, uint32_t timeout_ms)
{
    esp_err_t ret = ESP_OK;
    ret = esp_codec_dev_read(record_dev_handle, audio_buffer, len);
    *bytes_read = len;
    return ret;
}

esp_err_t product_board_codec_set_fs(esp_codec_dev_handle_t play_dev_handle, esp_codec_dev_handle_t record_dev_handle, uint32_t rate, uint32_t bits_cfg, i2s_slot_mode_t ch)
{
    esp_err_t ret = ESP_OK;

    esp_codec_dev_sample_info_t fs = {
        .sample_rate = rate,
        .channel = ch,
        .bits_per_sample = bits_cfg,
    };

    if (play_dev_handle) {
        ret = esp_codec_dev_close(play_dev_handle);
    }
    if (record_dev_handle) {
        ret |= esp_codec_dev_close(record_dev_handle);
        ret |= esp_codec_dev_set_in_gain(record_dev_handle, PRODUCT_CODEC_DEFAULT_ADC_VOLUME);
    }

    if (play_dev_handle) {
        ret |= esp_codec_dev_open(play_dev_handle, &fs);
    }
    if (record_dev_handle) {
        ret |= esp_codec_dev_open(record_dev_handle, &fs);
    }
    return ret;
}
