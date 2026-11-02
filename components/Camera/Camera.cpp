/* SPDX-License-Identifier: Apache-2.0 */
#include "Camera.hpp"
#include "panel_runtime.h"
#include "esp_heap_caps.h"
#include "bsp/esp-bsp.h"
extern "C" {
#include "common.h"
#include "provisioning.h"
}

LV_IMAGE_DECLARE(img_app_camera);
namespace esp_brookesia::apps {
Camera *Camera::_instance = nullptr;
Camera *Camera::requestInstance(bool status, bool navigation) {
    if (!_instance) _instance = new Camera(status, navigation);
    return _instance;
}
Camera::Camera(bool status, bool navigation) :
    App("Camera", &img_app_camera, true, status, navigation) {}

bool Camera::run() {
    if (_timer) return true;
    lv_obj_t *screen = lv_screen_active();
    lv_obj_set_style_bg_color(screen, lv_color_black(), 0);
    _image = lv_image_create(screen);
    lv_obj_center(_image);
    _status = lv_label_create(screen);
    lv_obj_set_style_text_color(_status, lv_color_white(), 0);
    lv_obj_align(_status, LV_ALIGN_TOP_MID, 0, 70);
    lv_label_set_text(_status, "Starting shared camera...");
    _frame = {};
    _frame.header.magic = LV_IMAGE_HEADER_MAGIC;
    _frame.header.cf = LV_COLOR_FORMAT_RGB565;
    _frame.data_size = 320 * 240 * 2;
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
        lv_image_set_scale(self->_image, 560 * 256 / self->_frame.header.w);
        lv_label_set_text(self->_status, provisioning_mode_active() ?
            "Scan a Wi-Fi / NyxID QR code. Back to exit." : "Camera · Back to exit");
    }
}
bool Camera::back() { return notifyCoreClosed(); }
bool Camera::close() {
    if (_timer) { lv_timer_delete(_timer); _timer = nullptr; }
    panel_runtime_set_preview(false);
    provisioning_mode_exit();
    if (_image) { lv_obj_delete(_image); _image = nullptr; }
    if (_status) { lv_obj_delete(_status); _status = nullptr; }
    lv_image_cache_drop(&_frame);
    heap_caps_free(const_cast<uint8_t *>(_frame.data));
    _frame = {};
    return true;
}
}
