#include "nyx_theme.h"
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
#include "panel_voice.h"
#include "lcd_text.h"
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
    lv_obj_set_style_bg_color(item, lv_color_hex(NYX_PRIMARY), 0);
    lv_obj_set_style_bg_color(item, lv_color_hex(NYX_PRESSED), LV_STATE_PRESSED);
    lv_obj_set_style_text_color(item, lv_color_hex(NYX_TEXT), 0);
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
    bots_revision = ~0u;
    login_ready = false;
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
    lv_obj_set_style_bg_color(root, lv_color_hex(NYX_SURFACE), 0);
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
        action = button(root, "Sign in / choose panel Key");
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
        label(root, "Voice Bot");
        voice_status = label(root, "");
        lv_obj_set_style_text_font(voice_status, lcd_text_font(LCD_TEXT_SIZE_MEDIUM), 0);
        bot_refresh = button(root, "Refresh available Bots");
        lv_obj_add_event_cb(bot_refresh, [](lv_event_t *) { (void)panel_voice_refresh_agents(); }, LV_EVENT_CLICKED, nullptr);
        bot_select = lv_dropdown_create(root);
        lv_obj_set_style_text_font(bot_select, lcd_text_font(LCD_TEXT_SIZE_MEDIUM), 0);
        lv_obj_set_style_text_font(lv_dropdown_get_list(bot_select), lcd_text_font(LCD_TEXT_SIZE_MEDIUM), 0);
        lv_obj_set_width(bot_select, lv_pct(100));
        lv_dropdown_set_options(bot_select, "Refresh Bots first");
        lv_obj_add_state(bot_select, LV_STATE_DISABLED);
        lv_obj_add_event_cb(bot_select, [](lv_event_t *event) {
            auto self = static_cast<PanelPage *>(lv_event_get_user_data(event));
            size_t index = lv_dropdown_get_selected(self->bot_select);
            if (index > 0 && index <= self->bot_count)
                (void)panel_voice_configure(self->bots[index - 1].agent_id);
        }, LV_EVENT_VALUE_CHANGED, this);
        node_setup = button(root, "Set up remote control connection");
        lv_obj_add_event_cb(node_setup, [](lv_event_t *) { (void)device_code_start_node_setup(); }, LV_EVENT_CLICKED, nullptr);
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
        lv_label_set_text_fmt(summary,
            "Device\n%s\n\nFirmware\n%s\n\nNetwork\n%s\n\nNyxID\n%s\n\nCamera\n%s\n\nUpdate\n%s",
            devices_self()->label, esp_app_get_description()->version, ip, connection,
            media_mode_is_preview() ? "Preview / scanning" : "Remote photos on request",
            panel_ota_busy() ? "Installing update" : panel_ota_error() != ESP_OK ? "Last update failed" : "Ready");
        return;
    }
    device_login_snapshot_t state{};
    device_code_snapshot(&state);
    if (state.state == DEVICE_LOGIN_IDLE) {
        lv_label_set_text_fmt(summary, "Remote access: %s\n%s\n\nPanel Key: %s\nSign in to select a Key for this panel, then choose your voice Bot below.",
            connection, nyx_creds_service_slug(), nyx_creds_have_agent() ? "Saved" : "Not signed in");
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
    lv_label_set_text(action_text, state.busy ? "Cancel sign-in" : "Sign in / choose panel Key");
    if (state.state == DEVICE_LOGIN_READY && state.node_setup) {
        lv_obj_remove_flag(restart, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(action, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(restart, LV_OBJ_FLAG_HIDDEN);
        lv_obj_remove_flag(action, LV_OBJ_FLAG_HIDDEN);
    }
    if (state.busy) lv_obj_add_state(node_setup, LV_STATE_DISABLED);
    else lv_obj_remove_state(node_setup, LV_STATE_DISABLED);
    if (state.state == DEVICE_LOGIN_READY && !state.node_setup && !login_ready) {
        login_ready = true;
        (void)panel_voice_refresh_agents();
    } else if (state.state != DEVICE_LOGIN_READY) login_ready = false;
    panel_voice_snapshot_t voice{};
    panel_voice_snapshot(&voice);
    lv_label_set_text_fmt(voice_status, "%s\n%s", voice.agent_name[0] ? voice.agent_name : "No Bot selected", voice.message);
    if (voice.busy || !nyx_creds_have_agent()) lv_obj_add_state(bot_refresh, LV_STATE_DISABLED);
    else lv_obj_remove_state(bot_refresh, LV_STATE_DISABLED);
    if (voice.bots_revision != bots_revision) {
        bots_revision = voice.bots_revision;
        bot_count = panel_voice_bots(bots, PANEL_VOICE_MAX_BOTS);
        lv_dropdown_clear_options(bot_select);
        lv_dropdown_add_option(bot_select, "Choose a Bot", LV_DROPDOWN_POS_LAST);
        size_t selected = 0;
        for (size_t i = 0; i < bot_count; ++i) {
            lv_dropdown_add_option(bot_select, bots[i].name, LV_DROPDOWN_POS_LAST);
            if (!strcmp(bots[i].agent_id, voice.agent_id)) selected = i + 1;
        }
        lv_dropdown_set_selected(bot_select, selected);
    }
    if (!bot_count || voice.busy) lv_obj_add_state(bot_select, LV_STATE_DISABLED);
    else lv_obj_remove_state(bot_select, LV_STATE_DISABLED);
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
