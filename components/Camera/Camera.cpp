#include "nyx_theme.h"
/* SPDX-License-Identifier: Apache-2.0 */
#include "Camera.hpp"
#include "panel_runtime.h"
#include "esp_heap_caps.h"
#include "bsp/esp-bsp.h"
extern "C" {
#include "common.h"
#include "provisioning.h"
#include "qr_scanner.h"
#include "lcd_renderer.h"
#include "lcd_text.h"
}
#include <cstdio>

LV_IMAGE_DECLARE(img_app_camera);
namespace esp_brookesia::apps {
Camera *Camera::_instance = nullptr;
Camera *Camera::requestInstance(bool status, bool navigation) {
    if (!_instance) _instance = new Camera(status, navigation);
    return _instance;
}
Camera::Camera(bool status, bool navigation) :
    App("Camera", &img_app_camera, true, status, navigation) {}

void Camera::setScanMode(bool enabled, const char *return_target) {
    _scanning = enabled;
    snprintf(_return_target, sizeof(_return_target), "%s", return_target);
    if (enabled) qr_scanner_begin();
    else { qr_scanner_cancel(); provisioning_mode_exit(); }
}

bool Camera::run() {
    if (_timer) return true;
    lv_obj_t *screen = lv_screen_active();
    lv_obj_set_style_bg_color(screen, lv_color_hex(NYX_BG), 0);
    _image = lv_image_create(screen);
    lv_obj_set_size(_image, 640, 880);
    lv_image_set_inner_align(_image, LV_IMAGE_ALIGN_CONTAIN);
    lv_obj_center(_image);
    _status = lv_label_create(screen);
    lv_obj_set_width(_status, BSP_LCD_H_RES - 80);
    lv_label_set_long_mode(_status, LV_LABEL_LONG_MODE_WRAP);
    lv_obj_set_style_text_color(_status, lv_color_white(), 0);
    lv_obj_set_style_text_font(_status, lcd_text_font(LCD_TEXT_SIZE_MEDIUM), 0);
    lv_obj_align(_status, LV_ALIGN_TOP_MID, 0, 70);
    lv_label_set_text(_status, "Starting shared camera...");
    _scan_action = lv_button_create(screen);
    lv_obj_set_size(_scan_action, 620, 72);
    lv_obj_align(_scan_action, LV_ALIGN_BOTTOM_MID, 0, -120);
    _scan_action_text = lv_label_create(_scan_action);
    lv_obj_center(_scan_action_text);
    lv_obj_add_flag(_scan_action, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_event_cb(_scan_action, [](lv_event_t *event) {
        auto self = static_cast<Camera *>(lv_event_get_user_data(event));
        qr_scan_snapshot_t state{};
        qr_scanner_snapshot(&state);
        if (state.state == QR_SCAN_CONFIRM) qr_scanner_confirm();
        else if (state.state == QR_SCAN_CONNECTED) self->back();
        else qr_scanner_begin();
    }, LV_EVENT_CLICKED, this);
    _exit_button = lv_button_create(screen);
    lv_obj_set_size(_exit_button, 620, 72);
    lv_obj_align(_exit_button, LV_ALIGN_BOTTOM_MID, 0, -24);
    auto exit_label = lv_label_create(_exit_button);
    lv_label_set_text(exit_label, "Back");
    lv_obj_center(exit_label);
    lv_obj_add_event_cb(_exit_button, [](lv_event_t *event) {
        static_cast<Camera *>(lv_event_get_user_data(event))->back();
    }, LV_EVENT_CLICKED, this);
    _frame = {};
    _frame.header.magic = LV_IMAGE_HEADER_MAGIC;
    _frame.header.cf = LV_COLOR_FORMAT_RGB565;
    _frame.data_size = PANEL_CAMERA_FRAME_BYTES;
    _frame.data = static_cast<uint8_t *>(heap_caps_malloc(_frame.data_size, MALLOC_CAP_SPIRAM));
    if (!_frame.data) {
        lv_label_set_text(_status, "Camera: insufficient memory");
        return true;
    }
    _sequence = 0;
    panel_runtime_set_preview(true); // The service worker performs CSI transitions outside LVGL.
    _timer = lv_timer_create(tick, 80, this);
    return true;
}
void Camera::tick(lv_timer_t *timer) {
    auto *self = static_cast<Camera *>(lv_timer_get_user_data(timer));
    if (self->_scanning) {
        qr_scan_snapshot_t state{};
        qr_scanner_snapshot(&state);
        lv_label_set_text_fmt(self->_status, "%s%s%s", state.message,
            state.ssid[0] ? "\n" : "", state.ssid);
        if (state.state == QR_SCAN_CONFIRM || state.state == QR_SCAN_ERROR || state.state == QR_SCAN_CONNECTED) {
            lv_label_set_text(self->_scan_action_text, state.state == QR_SCAN_CONFIRM ? "Connect to this network" :
                state.state == QR_SCAN_CONNECTED ? "Return to Settings" : "Scan again");
            lv_obj_remove_flag(self->_scan_action, LV_OBJ_FLAG_HIDDEN);
        } else lv_obj_add_flag(self->_scan_action, LV_OBJ_FLAG_HIDDEN);
    } else lv_obj_add_flag(self->_scan_action, LV_OBJ_FLAG_HIDDEN);
    if (!panel_runtime_preview_ready()) {
        esp_err_t err = panel_runtime_preview_error();
        if (err != ESP_OK) lv_label_set_text_fmt(self->_status, "Camera: %s", esp_err_to_name(err));
        return;
    }
    uint16_t w = 0, h = 0;
    size_t size = 0;
    uint32_t seq = 0;
    // Remove the widget's reference before changing its descriptor or pixel memory.
    lv_image_set_src(self->_image, nullptr);
    esp_err_t err = media_sys_preview_copy_rgb565(const_cast<uint8_t *>(self->_frame.data),
        self->_frame.data_size, self->_sequence, &w, &h, &size, &seq);
    if (err == ESP_OK) {
        self->_sequence = seq;
        self->_frame.header.w = w;
        self->_frame.header.h = h;
        self->_frame.header.stride = w * 2;
        lv_image_cache_drop(&self->_frame);
    }
    if (self->_frame.header.w) {
        lv_image_set_src(self->_image, &self->_frame);
        if (!self->_scanning) lv_label_set_text(self->_status, "Camera - Back to exit");
    }
}
bool Camera::back() {
    if (_scanning) {
        qr_scanner_cancel();
        return lcd_renderer_navigate(_return_target) == ESP_OK;
    }
    return notifyCoreClosed();
}
bool Camera::close() {
    if (_timer) { lv_timer_delete(_timer); _timer = nullptr; }
    panel_runtime_set_preview(false);
    qr_scanner_cancel();
    _scanning = false;
    provisioning_mode_exit();
    if (_image) { lv_obj_delete(_image); _image = nullptr; }
    if (_status) { lv_obj_delete(_status); _status = nullptr; }
    if (_scan_action) { lv_obj_delete(_scan_action); _scan_action = nullptr; _scan_action_text = nullptr; }
    if (_exit_button) { lv_obj_delete(_exit_button); _exit_button = nullptr; }
    lv_image_cache_drop(&_frame);
    heap_caps_free(const_cast<uint8_t *>(_frame.data));
    _frame = {};
    return true;
}
}
