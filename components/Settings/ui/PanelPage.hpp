#pragma once
#include "lvgl.h"
#include "panel_voice_proto.h"

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
    lv_obj_t *node_setup = nullptr;
    lv_obj_t *bot_select = nullptr;
    lv_obj_t *bot_refresh = nullptr;
    lv_obj_t *voice_status = nullptr;
    panel_voice_bot_t bots[PANEL_VOICE_MAX_BOTS]{};
    size_t bot_count = 0;
    unsigned bots_revision = ~0u;
    bool login_ready = false;
    lv_timer_t *timer = nullptr;
    char qr_value[384]{};
};
}
