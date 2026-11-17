/*
 * SPDX-FileCopyrightText: 2023-2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "MusicPlayer.hpp"
#include "lvgl.h"
#include "esp_brookesia.hpp"
#ifdef ESP_UTILS_LOG_TAG
#undef ESP_UTILS_LOG_TAG
#endif
#define ESP_UTILS_LOG_TAG "BS:MusicPlayer"
#include "esp_lib_utils.h"
#include "sdkconfig.h"
#include "bsp/esp-bsp.h"
#include "product_support.h"
#include "gui_music/lv_demo_music.h"
#include "gui_music/lv_demo_music_main.h"

#define MUSIC_DIR BSP_SPIFFS_MOUNT_POINT "/music"

LV_IMG_DECLARE(img_app_musicplayer);

static const char *TAG = "MusicPlayer";

namespace esp_brookesia::apps
{

    MusicPlayer *MusicPlayer::_instance = nullptr;

    MusicPlayer *MusicPlayer::requestInstance(bool use_status_bar, bool use_navigation_bar)
    {
        if (_instance == nullptr)
        {
            _instance = new MusicPlayer(use_status_bar, use_navigation_bar);
        }
        return _instance;
    }

    MusicPlayer::MusicPlayer(bool use_status_bar, bool use_navigation_bar) : App("MusicPlayer", &img_app_musicplayer, true, use_status_bar, use_navigation_bar),
                                                                             _file_iterator(NULL)
    {
    }

    MusicPlayer::~MusicPlayer()
    {
    }

    bool MusicPlayer::run(void)
    {
        ESP_UTILS_LOGD("Run");

        if (product_player_init() != ESP_OK)
        {
            ESP_LOGE(TAG, "Play init with SPIFFS failed");
            return false;
        }
        _player_ready = true;

        // The singleton reuses its immutable file list on subsequent launches.
        if (_file_iterator == nullptr && product_file_instance_init(MUSIC_DIR, &_file_iterator) != ESP_OK)
        {
            ESP_LOGE(TAG, "product_file_instance_init failed");
            product_player_del();
            _player_ready = false;
            return false;
        }

        lv_demo_music(lv_scr_act(), _file_iterator);
        return true;
    }

    bool MusicPlayer::back(void)
    {
        ESP_UTILS_LOGD("Back");
        // If the app needs to exit, call notifyCoreClosed() to notify the core to close the app
        ESP_UTILS_CHECK_FALSE_RETURN(notifyCoreClosed(), false, "Notify core closed failed");
        return true;
    }

    bool MusicPlayer::close(void)
    {
        ESP_UTILS_LOGD("Close");
        if (_player_ready && product_player_del() != ESP_OK)
        {
            ESP_LOGE(TAG, "DEL Play init with SPIFFS failed");
            return false;
        }
        _player_ready = false;
        return true;
    }

    bool MusicPlayer::init()
    {
        ESP_UTILS_LOGD("Init");
        return true;
    }

    bool MusicPlayer::deinit()
    {
        ESP_UTILS_LOGD("Deinit");
        return true;
    }

    bool MusicPlayer::pause()
    {
        ESP_UTILS_LOGD("Pause");
        if (!_player_ready) return true;
        // A paused app remains registered with Brookesia.  Pause the player
        // and release its shared audio owner so Nyx Panel can use the codec.
        lv_demo_music_pause();
        lv_demo_music_exit_pause();
        if (product_player_del() != ESP_OK)
        {
            ESP_LOGE(TAG, "DEL player while pausing failed");
            return false;
        }
        _player_ready = false;
        return true;
    }

    bool MusicPlayer::resume()
    {
        ESP_UTILS_LOGD("Resume");
        if (_player_ready) return true;
        if (product_player_init() != ESP_OK)
        {
            ESP_LOGE(TAG, "Play init while resuming failed");
            return false;
        }
        _player_ready = true;
        // Recreate the player and restore the selected track. The UI was left
        // in its paused state while the app was hidden, so resume also restores
        // the play button's checked state.
        lv_demo_music_resume();
        return true;
    }

} // namespace esp_brookesia::apps
