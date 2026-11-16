/*
 * SPDX-FileCopyrightText: 2015-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <string.h>
#include "esp_log.h"
#include "esp_check.h"
#include "esp_codec_dev_defaults.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/i2c.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "driver/ledc.h"

#include "bsp/esp-bsp.h"
#include "product_support.h"
#include "product_board.h"
#include "freertos/semphr.h"

static const char *TAG = "product_support";

static esp_codec_dev_handle_t play_dev_handle;
static esp_codec_dev_handle_t record_dev_handle;

static bool _is_audio_init = false;
static bool _is_player_init = false;
static SemaphoreHandle_t audio_owner;
static _Atomic(product_audio_yield_fn_t) background_yield;

esp_err_t product_audio_claim(void)
{
    if (!audio_owner) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(audio_owner,0)==pdTRUE) return ESP_OK;
    product_audio_yield_fn_t yield=atomic_load(&background_yield);
    return yield && yield() && xSemaphoreTake(audio_owner,pdMS_TO_TICKS(1000))==pdTRUE ?
        ESP_OK : ESP_ERR_INVALID_STATE;
}

esp_err_t product_audio_claim_background(product_audio_yield_fn_t yield)
{
    if (!yield || !audio_owner || xSemaphoreTake(audio_owner,0)!=pdTRUE) return ESP_ERR_INVALID_STATE;
    atomic_store(&background_yield,yield);
    return ESP_OK;
}

void product_audio_release(void)
{
    atomic_store(&background_yield,NULL);
    if (audio_owner) xSemaphoreGive(audio_owner);
}
static int _vloume_intensity = PRODUCT_CODEC_DEFAULT_VOLUME;

static audio_player_cb_t audio_idle_callback = NULL;
static void *audio_idle_cb_user_data = NULL;
static char audio_file_path[128];


/**************************************************************************************************
 *
 * Extra Board Function
 *
 **************************************************************************************************/

static esp_err_t audio_mute_function(AUDIO_PLAYER_MUTE_SETTING setting)
{
    // Volume saved when muting and restored when unmuting. Restoring volume is necessary
    // as es8311_set_voice_mute(true) results in voice volume (REG32) being set to zero.

    product_codec_mute_set(setting == AUDIO_PLAYER_MUTE ? true : false);

    // restore the voice volume upon unmuting
    if (setting == AUDIO_PLAYER_UNMUTE) {
        ESP_RETURN_ON_ERROR(esp_codec_dev_set_out_vol(play_dev_handle, _vloume_intensity), TAG, "Set Codec volume failed");
    }

    return ESP_OK;
}

static void audio_callback(audio_player_cb_ctx_t *ctx)
{
    if (audio_idle_callback) {
        ctx->user_ctx = audio_idle_cb_user_data;
        audio_idle_callback(ctx);
    }
}

int product_display_brightness_get(void)
{
    return product_board_display_brightness_get();
}

esp_err_t product_i2s_read(void *audio_buffer, size_t len, size_t *bytes_read, uint32_t timeout_ms)
{
    return product_board_i2s_read(record_dev_handle, audio_buffer, len, bytes_read, timeout_ms);
}

esp_err_t product_i2s_write(void *audio_buffer, size_t len, size_t *bytes_written, uint32_t timeout_ms)
{
    esp_err_t ret = ESP_OK;
    ret = esp_codec_dev_write(play_dev_handle, audio_buffer, len);
    *bytes_written = len;
    return ret;
}

esp_err_t product_codec_set_fs(uint32_t rate, uint32_t bits_cfg, i2s_slot_mode_t ch)
{
    return product_board_codec_set_fs(play_dev_handle, record_dev_handle, rate, bits_cfg, ch);
}

esp_err_t product_codec_volume_set(int volume, int *volume_set)
{
    ESP_RETURN_ON_ERROR(esp_codec_dev_set_out_vol(play_dev_handle, volume), TAG, "Set Codec volume failed");
    _vloume_intensity = volume;

    ESP_LOGI(TAG, "Setting volume: %d", volume);

    return ESP_OK;
}

int product_codec_volume_get(void)
{
    return _vloume_intensity;
}

esp_err_t product_codec_mute_set(bool enable)
{
    esp_err_t ret = ESP_OK;
    ret = esp_codec_dev_set_out_mute(play_dev_handle, enable);
    // ES8311 mute/reopen resets its output attenuation. Match MusicPlayer's
    // unmute path so the Settings volume also applies to direct voice playback.
    if (ret == ESP_OK && !enable)
        ret = esp_codec_dev_set_out_vol(play_dev_handle, _vloume_intensity);
    return ret;
}

esp_err_t product_codec_dev_stop(void)
{
    esp_err_t ret = ESP_OK;

    if (play_dev_handle) {
        ret = esp_codec_dev_close(play_dev_handle);
    }

    if (record_dev_handle) {
        ret = esp_codec_dev_close(record_dev_handle);
    }
    return ret;
}

esp_err_t product_codec_dev_resume(void)
{
    return product_codec_set_fs(PRODUCT_CODEC_DEFAULT_SAMPLE_RATE, PRODUCT_CODEC_DEFAULT_BIT_WIDTH, PRODUCT_CODEC_DEFAULT_CHANNEL);
}

esp_err_t product_codec_init(void)
{
    if (_is_audio_init) {
        return ESP_OK;
    }

    audio_owner = xSemaphoreCreateBinary();
    if (!audio_owner) return ESP_ERR_NO_MEM;
    xSemaphoreGive(audio_owner);
    play_dev_handle = bsp_audio_codec_speaker_init();
    assert((play_dev_handle) && "play_dev_handle not initialized");

    record_dev_handle = bsp_audio_codec_microphone_init();
    assert((record_dev_handle) && "record_dev_handle not initialized");

    ESP_RETURN_ON_ERROR(product_codec_set_fs(PRODUCT_CODEC_DEFAULT_SAMPLE_RATE, PRODUCT_CODEC_DEFAULT_BIT_WIDTH,
                                           PRODUCT_CODEC_DEFAULT_CHANNEL), TAG, "Open codecs failed");

    _is_audio_init = true;

    return ESP_OK;
}

esp_err_t product_player_init(void)
{
    if (_is_player_init) {
        return ESP_OK;
    }
    ESP_RETURN_ON_ERROR(product_audio_claim(), TAG, "Audio is busy");

    audio_player_config_t config = { .mute_fn = audio_mute_function,
                                     .write_fn = product_i2s_write,
                                     .clk_set_fn = product_codec_set_fs,
                                     .priority = 5
                                   };
    esp_err_t err = audio_player_new(config);
    if (err != ESP_OK) { product_audio_release(); return err; }
    audio_player_callback_register(audio_callback, NULL);

    _is_player_init = true;

    return ESP_OK;
}

esp_err_t product_player_del(void)
{
    if (!_is_player_init) return ESP_OK;
    ESP_RETURN_ON_ERROR(audio_player_delete(), TAG, "audio_player_delete failed");
    _is_player_init = false;
    product_audio_release();
    return ESP_OK;
}

esp_err_t product_file_instance_init(const char *path, file_iterator_instance_t **ret_instance)
{
    ESP_RETURN_ON_FALSE(path, ESP_FAIL, TAG, "path is NULL");
    ESP_RETURN_ON_FALSE(ret_instance, ESP_FAIL, TAG, "ret_instance is NULL");

    file_iterator_instance_t *file_iterator = file_iterator_new(path);
    ESP_RETURN_ON_FALSE(file_iterator, ESP_FAIL, TAG, "file_iterator_new failed, %s", path);

    *ret_instance = file_iterator;

    return ESP_OK;
}

esp_err_t product_player_play_index(file_iterator_instance_t *instance, int index)
{
    ESP_RETURN_ON_FALSE(instance, ESP_FAIL, TAG, "instance is NULL");

    ESP_LOGI(TAG, "play_index(%d)", index);
    char filename[128];
    int retval = file_iterator_get_full_path_from_index(instance, index, filename, sizeof(filename));
    ESP_RETURN_ON_FALSE(retval != 0, ESP_FAIL, TAG, "file_iterator_get_full_path_from_index failed");

    ESP_LOGI(TAG, "opening file '%s'", filename);
    FILE *fp = fopen(filename, "rb");
    ESP_RETURN_ON_FALSE(fp, ESP_FAIL, TAG, "unable to open file");

    ESP_LOGI(TAG, "Playing '%s'", filename);
    ESP_RETURN_ON_ERROR(audio_player_play(fp), TAG, "audio_player_play failed");

    memcpy(audio_file_path, filename, sizeof(audio_file_path));

    return ESP_OK;
}

esp_err_t product_player_play_file(const char *file_path)
{
    ESP_LOGI(TAG, "opening file '%s'", file_path);
    FILE *fp = fopen(file_path, "rb");
    ESP_RETURN_ON_FALSE(fp, ESP_FAIL, TAG, "unable to open file");

    ESP_LOGI(TAG, "Playing '%s'", file_path);
    ESP_RETURN_ON_ERROR(audio_player_play(fp), TAG, "audio_player_play failed");

    memcpy(audio_file_path, file_path, sizeof(audio_file_path));

    return ESP_OK;
}

void product_player_register_callback(audio_player_cb_t cb, void *user_data)
{
    audio_idle_callback = cb;
    audio_idle_cb_user_data = user_data;
}

bool product_player_is_playing_by_path(const char *file_path)
{
    return (strcmp(audio_file_path, file_path) == 0);
}

bool product_player_is_playing_by_index(file_iterator_instance_t *instance, int index)
{
    return (index == file_iterator_get_index(instance));
}

esp_codec_dev_handle_t product_codec_speaker(void) { return play_dev_handle; }
esp_codec_dev_handle_t product_codec_microphone(void) { return record_dev_handle; }
