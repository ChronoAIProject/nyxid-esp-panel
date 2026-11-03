#include "PanelPage.hpp"
#include "../Settings.hpp"
#include "esp_app_desc.h"
#include "esp_system.h"
extern "C" {
#include "device_code.h"
#include "devices.h"
#include "lcd_renderer.h"
#include "media_mode.h"
#include "net_manager.h"
#include "nyx_creds.h"
#include "nyx_node.h"
#include "panel_ota.h"
#include "stream_publisher.h"
}
#include <cstring>

namespace esp_brookesia::apps {
namespace {
lv_obj_t *label(lv_obj_t *parent, const char *text)
{
    auto item = lv_label_create(parent);
    lv_obj_set_width(item, lv_pct(100));
    lv_label_set_long_mode(item, LV_LABEL_LONG_MODE_WRAP);
    lv_label_set_text(item, text);
    return item;
}
lv_obj_t *button(lv_obj_t *parent, const char *text)
{
    auto item = lv_button_create(parent);
    lv_obj_set_width(item, lv_pct(100));
    lv_obj_set_height(item, 72);
    auto caption = lv_label_create(item);
    lv_label_set_text(caption, text);
    lv_obj_center(caption);
    return item;
}
}

void PanelPage::run(bool show_nyx)
{
    close();
    nyx = show_nyx;
    root = lv_obj_create(lv_screen_active());
    lv_obj_add_event_cb(root, [](lv_event_t *event) {
        auto self = static_cast<PanelPage *>(lv_event_get_user_data(event));
        if (self->timer) { lv_timer_delete(self->timer); self->timer = nullptr; }
        if (self->nyx) device_code_cancel();
        self->root = nullptr;
        self->qr_value[0] = 0;
    }, LV_EVENT_DELETE, this);
    lv_obj_set_size(root, BSP_LCD_H_RES - 80, BSP_LCD_V_RES - 160);
    lv_obj_center(root);
    lv_obj_set_style_bg_color(root, lv_color_hex(0x101114), 0);
    lv_obj_set_style_text_color(root, lv_color_white(), 0);
    lv_obj_set_style_text_font(root, &lv_font_montserrat_30, 0);
    lv_obj_set_style_border_width(root, 0, 0);
    lv_obj_set_style_pad_all(root, 24, 0);
    lv_obj_set_style_pad_row(root, 22, 0);
    lv_obj_set_flex_flow(root, LV_FLEX_FLOW_COLUMN);
    auto back = button(root, LV_SYMBOL_LEFT "  Settings");
    lv_obj_add_event_cb(back, [](lv_event_t *) {
        lv_async_call([](void *) { Settings::requestInstance()->showRoot(); }, nullptr);
    }, LV_EVENT_CLICKED, nullptr);
    label(root, nyx ? "NyxID" : "Device status");
    summary = label(root, "");
    if (nyx) {
        code = label(root, "");
        lv_obj_set_style_text_align(code, LV_TEXT_ALIGN_CENTER, 0);
        qr = lv_qrcode_create(root);
        lv_qrcode_set_size(qr, 300);
        lv_qrcode_set_dark_color(qr, lv_color_black());
        lv_qrcode_set_light_color(qr, lv_color_white());
        lv_obj_set_style_pad_all(qr, 20, 0);
        lv_obj_set_style_bg_color(qr, lv_color_white(), 0);
        lv_obj_set_style_bg_opa(qr, LV_OPA_COVER, 0);
        lv_obj_set_style_align(qr, LV_ALIGN_CENTER, 0);
        remaining = label(root, "");
        action = button(root, "Sign in to NyxID");
        action_text = lv_obj_get_child(action, 0);
        lv_obj_add_event_cb(action, [](lv_event_t *event) {
            auto self = static_cast<PanelPage *>(lv_event_get_user_data(event));
            device_login_snapshot_t state{};
            device_code_snapshot(&state);
            if (state.busy) device_code_cancel();
            else (void)device_code_start_if_needed();
            self->refresh();
        }, LV_EVENT_CLICKED, this);
        restart = button(root, "Restart to activate connection");
        lv_obj_add_event_cb(restart, [](lv_event_t *) { esp_restart(); }, LV_EVENT_CLICKED, nullptr);
        auto scan = button(root, "Scan Wi-Fi QR code");
        lv_obj_add_event_cb(scan, [](lv_event_t *) { lcd_renderer_navigate("scan"); }, LV_EVENT_CLICKED, nullptr);
    }
    refresh();
    timer = lv_timer_create([](lv_timer_t *timer) {
        static_cast<PanelPage *>(lv_timer_get_user_data(timer))->refresh();
    }, 500, this);
}

void PanelPage::refresh()
{
    if (!root) return;
    auto net = net_manager_snapshot();
    const char *ip = net.eth_ip ? net.eth_ip_addr : net.wifi_ip ? net.wifi_ip_addr : "Not connected";
    const char *connection = nyx_node_state() == NYX_NODE_UP ? "Connected" :
        nyx_node_state() == NYX_NODE_CONNECTING ? "Connecting" : "Not connected";
    if (!nyx) {
        stream_publisher_status_t stream{};
        stream_publisher_get_status(&stream);
        lv_label_set_text_fmt(summary,
            "Device\n%s\n\nFirmware\n%s\n\nNetwork\n%s\n\nNyxID\n%s\n\nCamera\n%s\n\nVideo stream\n%s\n\nUpdate\n%s",
            devices_self()->label, esp_app_get_description()->version, ip, connection,
            media_mode_is_preview() ? "Preview / scanning" : "Available for streaming",
            !strcmp(stream.state, "idle") ? "Stopped" : stream.state,
            panel_ota_busy() ? "Installing update" : panel_ota_error() != ESP_OK ? "Last update failed" : "Ready");
        return;
    }
    device_login_snapshot_t state{};
    device_code_snapshot(&state);
    if (state.state == DEVICE_LOGIN_IDLE) {
        lv_label_set_text_fmt(summary, "Remote access: %s\n%s\n\nSign in on your phone to connect this panel. Your current connection stays available until setup is complete.",
            connection, nyx_creds_service_slug());
    } else lv_label_set_text(summary, state.message);
    bool waiting = state.state == DEVICE_LOGIN_WAITING;
    for (auto obj : {code, qr, remaining}) {
        if (waiting) lv_obj_remove_flag(obj, LV_OBJ_FLAG_HIDDEN);
        else lv_obj_add_flag(obj, LV_OBJ_FLAG_HIDDEN);
    }
    if (waiting) {
        lv_label_set_text(code, state.user_code);
        lv_label_set_text_fmt(remaining, "%s\nExpires in %lu:%02lu", state.verification_uri,
            (unsigned long)(state.seconds_remaining / 60), (unsigned long)(state.seconds_remaining % 60));
        if (strcmp(qr_value, state.verification_uri_complete)) {
            if (lv_qrcode_update(qr, state.verification_uri_complete, strlen(state.verification_uri_complete)) == LV_RESULT_OK)
                strcpy(qr_value, state.verification_uri_complete);
        }
    }
    lv_label_set_text(action_text, state.busy ? "Cancel sign-in" : "Sign in to NyxID");
    if (state.state == DEVICE_LOGIN_READY) {
        lv_obj_remove_flag(restart, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(action, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(restart, LV_OBJ_FLAG_HIDDEN);
        lv_obj_remove_flag(action, LV_OBJ_FLAG_HIDDEN);
    }
}

void PanelPage::close()
{
    if (timer) { lv_timer_delete(timer); timer = nullptr; }
    if (root) {
        if (nyx) device_code_cancel();
        lv_obj_delete(root);
        root = nullptr;
    }
    qr_value[0] = 0;
}
}
