#pragma once
#include "systems/phone/esp_brookesia_phone_app.hpp"
#include "systems/phone/esp_brookesia_phone.hpp"

namespace esp_brookesia::apps {
class NyxPanel : public systems::phone::App {
public:
    NyxPanel();
    static void attach(systems::phone::Phone *phone, NyxPanel *app);
    void render();
    void refreshVoice();
    void showVoiceControls();
protected:
    bool run() override;
    bool back() override { return notifyCoreClosed(); }
    bool close() override;
    bool init() override { return true; }
    bool deinit() override { return true; }
    bool pause() override { return close(); }
    bool resume() override { return run(); }
private:
    lv_obj_t *root = nullptr;
    lv_obj_t *content = nullptr;
    lv_obj_t *voice_status = nullptr;
    lv_obj_t *call_button = nullptr;
    lv_obj_t *call_label = nullptr;
    lv_obj_t *hold_button = nullptr;
    lv_obj_t *mode_select = nullptr;
    lv_obj_t *voice_controls = nullptr;
    lv_obj_t *voice_toggle = nullptr;
    lv_obj_t *voice_indicator = nullptr;
};
}
