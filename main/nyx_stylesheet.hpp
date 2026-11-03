#pragma once

#include "systems/phone/stylesheets/800_1280/dark/stylesheet.hpp"
#include "nyx_theme.h"

// Keep the vendor's panel dimensions, typography and gestures; supply NyxID colors.
inline const esp_brookesia::systems::phone::Stylesheet &nyx_stylesheet()
{
    using namespace esp_brookesia;
    // The stylesheet is larger than the 4 KiB startup stack. Keep both its
    // storage and the returned value off the stack (including call-site copies).
    static auto sheet = systems::phone::STYLESHEET_800_1280_DARK;
    sheet.core.name = "NyxID 800x1280";
    sheet.core.display.background.color = gui::StyleColor::COLOR(NYX_BG);
    auto &display = sheet.display;
    display.status_bar.data.main.background_color = gui::StyleColor::COLOR(NYX_BG);
    display.status_bar.data.main.text_color = gui::StyleColor::COLOR(NYX_TINT);
    display.navigation_bar.data.main.background_color = gui::StyleColor::COLOR(NYX_SURFACE);
    display.navigation_bar.data.button.active_background_color = gui::StyleColor::COLOR(NYX_PRIMARY);
    display.app_launcher.data.indicator.spot_inactive_background_color = gui::StyleColor::COLOR(NYX_BORDER);
    display.app_launcher.data.indicator.spot_active_background_color = gui::StyleColor::COLOR(NYX_SECONDARY);
    display.recents_screen.data.main.background_color = gui::StyleColor::COLOR(NYX_BG);
    display.recents_screen.data.memory.label_text_color = gui::StyleColor::COLOR(NYX_MUTED);
    for (auto &bar : sheet.manager.gesture.indicator_bars) {
        bar.main.color = gui::StyleColor::COLOR(NYX_SURFACE);
        bar.indicator.color = gui::StyleColor::COLOR(NYX_SECONDARY);
    }
    return sheet;
}
