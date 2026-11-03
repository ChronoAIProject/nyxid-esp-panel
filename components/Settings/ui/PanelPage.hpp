#pragma once
#include "lvgl.h"

namespace esp_brookesia::apps {
class PanelPage {
public:
    void run(bool nyx);
    void close();
private:
    void refresh();
    bool nyx = false;
    lv_obj_t *root = nullptr;
    lv_obj_t *summary = nullptr;
    lv_obj_t *code = nullptr;
    lv_obj_t *qr = nullptr;
    lv_obj_t *remaining = nullptr;
    lv_obj_t *action = nullptr;
    lv_obj_t *action_text = nullptr;
    lv_obj_t *restart = nullptr;
    lv_timer_t *timer = nullptr;
    char qr_value[384]{};
};
}
