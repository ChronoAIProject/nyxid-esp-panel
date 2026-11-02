#pragma once
#include "systems/phone/esp_brookesia_phone_app.hpp"
#include "lvgl.h"

namespace esp_brookesia::apps {
class Camera : public systems::phone::App {
public:
    static Camera *requestInstance(bool use_status_bar = true, bool use_navigation_bar = true);
    ~Camera() override = default;
protected:
    Camera(bool status, bool navigation);
    bool run() override;
    bool back() override;
    bool close() override;
    bool init() override { return true; }
    bool deinit() override { return true; }
    bool pause() override { return close(); }
    bool resume() override { return run(); }
private:
    static Camera *_instance;
    lv_obj_t *_image = nullptr;
    lv_obj_t *_status = nullptr;
    lv_timer_t *_timer = nullptr;
    lv_image_dsc_t _frame = {};
    uint32_t _sequence = 0;
    static void tick(lv_timer_t *timer);
};
}
