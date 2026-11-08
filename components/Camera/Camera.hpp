#pragma once

#include "systems/phone/esp_brookesia_phone_app.hpp"
#include "bsp/esp-bsp.h"
#include "product_support.h"
#include "driver/ppa.h"
#include "esp_lcd_touch.h"
#include "esp_lcd_types.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lvgl.h"
#include "sdkconfig.h"
#include "camera_profile.h"
#include <atomic>

namespace esp_brookesia::apps {
class Camera : public systems::phone::App {
public:
    static Camera *requestInstance(bool use_status_bar = true, bool use_navigation_bar = true);
    ~Camera();
    void setScanMode(bool enabled, const char *return_target = "settings");
    bool scanning() const { return _scanning; }
    // Native preview pauses LVGL; wake it through task cleanup so queued
    // navigation can run on the LVGL thread after the camera has stopped.
    void stopPreviewForNavigation() { _preview_running = false; }
protected:
    Camera(bool use_status_bar, bool use_navigation_bar);
    bool run() override;
    bool back() override;
    bool close() override;
    bool init() override;
    bool deinit() override;
    bool pause() override;
    bool resume() override;
private:
    static Camera *_instance;
    // Native vendor preview uses PPA directly into the LCD scanout buffer.
    // Keep native Camera preview on the same profile as the shared camera
    // service/QR path.  These values are defined once in camera_profile.h.
    static constexpr int CAMERA_BUFFER_COUNT = PANEL_CAMERA_BUFFER_COUNT;
    static constexpr int PREVIEW_TASK_STACK_SIZE = PANEL_CAMERA_TASK_STACK;
    static constexpr int PREVIEW_TASK_PRIORITY = PANEL_CAMERA_TASK_PRIORITY;
    static constexpr int SWIPE_EXIT_THRESHOLD = 180;
    lv_obj_t *_status_label = nullptr;
    lv_obj_t *_exit_button = nullptr;
    lv_display_t *_display = nullptr;
    esp_lcd_panel_handle_t _display_panel = nullptr;
    ppa_client_handle_t _ppa_srm_handle = nullptr;
    esp_lcd_touch_handle_t _touch_handle = nullptr;
    TaskHandle_t _preview_task_handle = nullptr;
    TaskHandle_t _close_wait_task = nullptr;
    int _video_fd = -1;
    bool _video_driver_initialized = false;
    std::atomic_bool _preview_running{false};
    bool _dummy_enabled = false;
    bool _lvgl_paused = false;
    bool _touch_active = false;
    uint16_t _touch_start_x = 0, _touch_start_y = 0;
    uint32_t _camera_width = 0, _camera_height = 0;
    size_t _camera_buffer_size = 0;
    size_t _camera_buffer_lengths[CAMERA_BUFFER_COUNT] = {};
    size_t _data_cache_line_size = 64;
    void *_camera_buffers[CAMERA_BUFFER_COUNT] = {};
    void *_lcd_buffers[CONFIG_BSP_LCD_DPI_BUFFER_NUMS] = {};
    // QR/Wi-Fi scan path retains the bounded service-backed preview.
    lv_obj_t *_image = nullptr;
    lv_obj_t *_status = nullptr;
    lv_timer_t *_timer = nullptr;
    lv_image_dsc_t _frame = {};
    uint32_t _sequence = 0;
    bool _scanning = false;
    char _return_target[24] = "settings";
    lv_obj_t *_scan_action = nullptr;
    lv_obj_t *_scan_action_text = nullptr;

    bool startDirectPreview();
    void requestStopDirectPreview();
    esp_err_t initDirectVideoDriver();
    esp_err_t openDirectVideoDevice();
    esp_err_t setupDirectCameraBuffers();
    esp_err_t startDirectDummyPreview();
    void stopDirectDummyPreview();
    void releaseDirectCameraBuffers();
    bool shouldDirectExitBySwipe();
    esp_err_t handleDirectFrame();
    void directPreviewTask();
    static void directPreviewTaskEntry(void *arg);
    bool runScan();
    bool closeScan();
    static void tickScan(lv_timer_t *timer);
};
}
