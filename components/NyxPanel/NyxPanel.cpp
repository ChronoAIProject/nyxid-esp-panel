/* SPDX-License-Identifier: Apache-2.0 */
#include "NyxPanel.hpp"
#include "Settings.hpp"
#include "Camera.hpp"
#include "esp_brookesia.hpp"
#include "bsp/esp-bsp.h"
#include "esp_heap_caps.h"
#include "esp_cache.h"
#include "esp_timer.h"
#include "driver/jpeg_encode.h"
#include "esp_log.h"
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "lcd_renderer.h"
#include "lcd_image.h"
#include "lcd_background.h"
#include "nyx_theme.h"
#include "nvs.h"
#include "lcd_text.h"
#include "panel_dpi_compat.h"
#include "nvs_config.h"
#include "provisioning.h"
}
LV_IMAGE_DECLARE(nyx_panel_icon);
extern "C" { static esp_err_t navigate_now(const char *target); }
using esp_brookesia::apps::NyxPanel;
using esp_brookesia::systems::base::Context;
using esp_brookesia::systems::base::Manager;
using esp_brookesia::systems::phone::Phone;
namespace {
Phone *phone;
NyxPanel *app;
lv_obj_t *image_widget;
lcd_image_t image{};
int64_t image_expiry;
struct Message {
    char text[1025]{};
    int64_t expiry{};
    uint32_t color{0xffffff}, bg{NYX_BG};
    lcd_text_size_t size{LCD_TEXT_SIZE_MEDIUM};
    lcd_text_align_t align{LCD_TEXT_ALIGN_LEFT};
};
std::array<Message, 20> messages;
size_t count;
char note[160]{};
lv_timer_t *expiry_timer;
SemaphoreHandle_t screenshot_lock;
SemaphoreHandle_t background_lock;
SemaphoreHandle_t background_ready;
esp_err_t background_error;
lcd_image_t background{};
bool background_stored;
size_t background_bytes;
extern const uint8_t builtin_start[] asm("_binary_nyxbot_default_jpg_start");
extern const uint8_t builtin_end[] asm("_binary_nyxbot_default_jpg_end");
bool dirty;
char pending_navigation[4][24];
size_t navigation_count;

// The bundled Chinese font contains bitmap glyphs. LVGL's vector-font outline
// styles do not stroke them, so render the same wrapped text at eight offsets.
// Labels stay transparent and share the same bounds, including for long text.
void caption(lv_obj_t *parent, const Message &message) {
    constexpr int stroke = 2;
    constexpr int margin = 40;
    constexpr int bottom = 56; // Keep the native bottom-edge gesture accessible.
    lv_obj_update_layout(parent);
    int width = lv_obj_get_content_width(parent) - 2 * margin;
    int max_height = lv_obj_get_content_height(parent) - 2 * bottom;
    auto layer = [&](int x, int y, uint32_t color) {
        auto label = lv_label_create(parent);
        lv_obj_set_width(label, width);
        lv_obj_set_style_max_height(label, max_height, 0);
        lcd_text_style_t style{message.text, message.size, color, message.bg, message.align};
        lcd_text_apply(label, &style);
        lv_obj_set_style_bg_opa(label, LV_OPA_TRANSP, 0);
        lv_obj_align(label, LV_ALIGN_BOTTOM_MID, x, -bottom + y);
    };
    for (int y = -stroke; y <= stroke; y += stroke)
        for (int x = -stroke; x <= stroke; x += stroke)
            if (x || y) layer(x, y, message.bg);
    layer(0, 0, message.color);
}


esp_err_t activate(esp_brookesia::systems::base::App *target) {
    if (!phone || !target) return ESP_ERR_INVALID_STATE;
    if (phone->getManager().getActiveApp() != target) {
        Context::AppEventData event{target->getId(), Context::AppEventType::START, nullptr};
        if (!phone->sendAppEvent(&event)) return ESP_FAIL;
    }
    return phone->getManager().getActiveApp() == target ? ESP_OK : ESP_FAIL;
}
uint32_t parse_color(const char *s, uint32_t fallback) {
    if (!s || !*s) return fallback;
    if (*s == '#') ++s;
    if (strlen(s) != 6) return fallback;
    char *end = nullptr;
    unsigned long c = strtoul(s, &end, 16);
    return end && !*end ? c : fallback;
}
void remove_image() {
    if (image_widget) lv_image_set_src(image_widget, nullptr);
    lv_image_cache_drop(&image.dsc);
    lcd_image_free(&image);
    image_expiry = 0;
}
void expire(lv_timer_t *) {
    int64_t now = esp_timer_get_time();
    bool changed = false;
    if (image_expiry && now >= image_expiry) { remove_image(); changed = true; }
    for (size_t i = 0; i < count;) {
        if (messages[i].expiry && now >= messages[i].expiry) {
            for (size_t j = i + 1; j < count; ++j) messages[j - 1] = messages[j];
            --count;
            changed = true;
        } else ++i;
    }
    if (navigation_count) {
        char target[24];
        snprintf(target,sizeof(target),"%s",pending_navigation[0]);
        for (size_t i=1; i<navigation_count; ++i)
            memcpy(pending_navigation[i-1],pending_navigation[i],24);
        --navigation_count;
        esp_err_t err=navigate_now(target);
        if (err!=ESP_OK) ESP_LOGE("panel.gui","Native navigation failed: %s",esp_err_to_name(err));
    }
    if ((changed || dirty) && app) { dirty=false; app->render(); }
}
}
namespace esp_brookesia::apps {
NyxPanel::NyxPanel() : App("Nyx Panel", &nyx_panel_icon, true, false, true) {}
void NyxPanel::attach(Phone *p, NyxPanel *a) { phone = p; app = a; }
bool NyxPanel::run() {
    root = lv_obj_create(lv_screen_active());
    lv_obj_set_size(root, lv_pct(100), lv_pct(100));
    lv_obj_align(root, LV_ALIGN_TOP_LEFT, 0, 0);
    lv_obj_set_style_bg_color(root, lv_color_hex(NYX_BG), 0);
    lv_obj_set_style_bg_opa(root, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(root, 0, 0);
    lv_obj_set_style_radius(root, 0, 0);
    lv_obj_set_style_pad_all(root, 0, 0);
    lv_obj_remove_flag(root, LV_OBJ_FLAG_SCROLLABLE);
    render();
    return true;
}
bool NyxPanel::close() {
    image_widget = nullptr;
    if (root) { lv_obj_delete(root); root = nullptr; }
    return true;
}
void NyxPanel::render() {
    if (!root) return;
    image_widget = nullptr;
    lv_obj_clean(root);
    const auto &visible = image.dsc.data ? image : background;
    if (visible.dsc.data) {
        image_widget = lv_image_create(root);
        lv_image_set_src(image_widget, &visible.dsc);
        lv_obj_update_layout(root);
        // Round cover scaling up: LVGL's built-in COVER rounds down and can
        // leave a one-pixel seam on opposite edges for non-integral ratios.
        int sx = (lv_obj_get_content_width(root) * LV_SCALE_NONE + visible.width - 1) / visible.width;
        int sy = (lv_obj_get_content_height(root) * LV_SCALE_NONE + visible.height - 1) / visible.height;
        lv_image_set_scale(image_widget, sx > sy ? sx : sy);
        lv_obj_center(image_widget);
    }
    // Show only the latest unexpired caption. Keeping the bounded history lets
    // a temporary caption expire back to the previous one without hiding art.
    if (count) caption(root, messages[count - 1]);
    if (!count && !visible.dsc.data) {
        auto label = lv_label_create(root);
        lv_obj_set_style_text_font(label, lcd_text_font(LCD_TEXT_SIZE_MEDIUM), 0);
        lv_obj_set_style_text_color(label, lv_color_hex(NYX_SECONDARY), 0);
        lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_CENTER, 0);
        lv_label_set_text_fmt(label, "Nyx Panel\n%s", note[0] ? note : "Ready for messages and images");
        lv_obj_center(label);
    }
}
}
extern "C" {
esp_err_t lcd_renderer_init(void) {
    if (!screenshot_lock) screenshot_lock = xSemaphoreCreateMutex();
    if (!background_lock) background_lock = xSemaphoreCreateMutex();
    if (!expiry_timer) expiry_timer = lv_timer_create(expire, 250, nullptr);
    return screenshot_lock && background_lock && expiry_timer ? ESP_OK : ESP_ERR_NO_MEM;
}
static void prepare_background(void *) {
    // Decode on core 1 while startup initializes networking, codec and display.
    uint8_t *saved = nullptr;
    size_t len = 0;
    esp_err_t err = lcd_background_read(&saved, &len);
    if (err == ESP_OK) err = lcd_image_decode_jpeg(saved, len, &background);
    free(saved);
    background_stored = err == ESP_OK;
    if (!background_stored) {
        if (err != ESP_ERR_NVS_NOT_FOUND)
            ESP_LOGW("panel.gui", "Default image unavailable (%s); using built-in art", esp_err_to_name(err));
        len = builtin_end - builtin_start;
        err = lcd_image_decode_jpeg(builtin_start, len, &background);
    }
    background_error = err;
    background_bytes = len;
    ESP_LOGI("panel.gui", "Default image prepared: %s", esp_err_to_name(err));
    xSemaphoreGive(background_ready);
    vTaskDelete(nullptr);
}
esp_err_t lcd_renderer_prepare(void) {
    if (background_ready) return ESP_ERR_INVALID_STATE;
    background_ready = xSemaphoreCreateBinary();
    if (!background_ready) return ESP_ERR_NO_MEM;
    if (xTaskCreatePinnedToCore(prepare_background, "panel_art", 12288, nullptr, 1, nullptr, 1) != pdPASS) {
        vSemaphoreDelete(background_ready);
        background_ready = nullptr;
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
esp_err_t lcd_renderer_start(void) {
    if (!background_ready) return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(background_ready, pdMS_TO_TICKS(10000)) != pdTRUE) return ESP_ERR_TIMEOUT;
    vSemaphoreDelete(background_ready);
    background_ready = nullptr;
    if (background_error != ESP_OK) return background_error;
    return navigate_now("home");
}

static esp_err_t update_default_image(const uint8_t *jpeg, size_t len, bool reset) {
    if (!background_lock) return ESP_ERR_INVALID_STATE;
    if (!jpeg || !len || len > LCD_BACKGROUND_MAX_BYTES) return ESP_ERR_INVALID_SIZE;
    if (xSemaphoreTake(background_lock, pdMS_TO_TICKS(10000)) != pdTRUE) return ESP_ERR_TIMEOUT;
    lcd_image_t decoded{};
    esp_err_t err = lcd_image_decode_jpeg(jpeg, len, &decoded);
    // Validate fully before touching the persistent blob. Flash is written with
    // no GUI lock held; NVS atomically replaces the blob across power loss.
    if (err == ESP_OK) err = reset ? lcd_background_reset() : lcd_background_write(jpeg, len);
    if (err == ESP_OK) {
        // Commit succeeded; wait for the GUI so runtime always agrees with NVS.
        err = bsp_display_lock(static_cast<uint32_t>(-1));
        if (err == ESP_OK) {
            if (image_widget) lv_image_set_src(image_widget, nullptr);
            lv_image_cache_drop(&background.dsc);
            lcd_image_free(&background);
            background = decoded;
            decoded = {};
            background_stored = !reset;
            background_bytes = len;
            dirty = true;
            bsp_display_unlock();
        }
    }
    lcd_image_free(&decoded);
    xSemaphoreGive(background_lock);
    return err;
}
esp_err_t lcd_renderer_set_default_image(const uint8_t *jpeg, size_t len) {
    return update_default_image(jpeg, len, false);
}
esp_err_t lcd_renderer_reset_default_image(void) {
    return update_default_image(builtin_start, builtin_end - builtin_start, true);
}
esp_err_t lcd_renderer_default_image_info(lcd_default_image_info_t *info) {
    if (!info) return ESP_ERR_INVALID_ARG;
    if (bsp_display_lock(2000) != ESP_OK) return ESP_ERR_TIMEOUT;
    *info = {background_stored, background.width, background.height, background_bytes};
    bsp_display_unlock();
    return ESP_OK;
}
void lcd_renderer_set_net(const net_manager_snapshot_t *) {}
void lcd_renderer_refresh_status(void) {}
void lcd_renderer_reload_config(void) {
    cfg_lcd_t cfg{};
    if (nvs_config_get_lcd(&cfg) == ESP_OK) bsp_display_brightness_set(cfg.brightness);
}
void lcd_renderer_set_message(const char *text) {
    (void)lcd_renderer_push_text(text, "medium", "#ffffff", "#0d0d0d", "left", 0);
}
void lcd_renderer_note_qr_decoded(const char *kind) {
    if (bsp_display_lock(1000) != ESP_OK) return;
    snprintf(note, sizeof(note), "QR provisioning: %s", kind ? kind : "applied");
    dirty=true;
    bsp_display_unlock();
}
esp_err_t lcd_renderer_push_image(const uint8_t *jpeg, size_t len, uint32_t duration) {
    if (!jpeg || !len || len > 65536) return ESP_ERR_INVALID_SIZE;
    lcd_image_t decoded{};
    esp_err_t err = lcd_image_decode_jpeg(jpeg, len, &decoded);
    if (err != ESP_OK) return err;
    if (bsp_display_lock(2000) != ESP_OK) { lcd_image_free(&decoded); return ESP_ERR_TIMEOUT; }
    remove_image();
    image = decoded;
    image_expiry = duration ? esp_timer_get_time() + int64_t(duration) * 1000 : 0;
    err = lcd_renderer_navigate("home");
    dirty=true;
    bsp_display_unlock();
    return err;
}
esp_err_t lcd_renderer_push_text(const char *text, const char *size, const char *color,
                                  const char *bg, const char *align, uint32_t duration) {
    if (!text || !*text || strlen(text) > 1024) return ESP_ERR_INVALID_ARG;
    if (bsp_display_lock(2000) != ESP_OK) return ESP_ERR_TIMEOUT;
    if (count == messages.size()) {
        for (size_t i = 1; i < count; ++i) messages[i - 1] = messages[i];
        --count;
    }
    auto &msg = messages[count++];
    msg = {};
    snprintf(msg.text, sizeof(msg.text), "%s", text);
    msg.expiry = duration ? esp_timer_get_time() + int64_t(duration) * 1000 : 0;
    msg.color = parse_color(color, 0xffffff);
    msg.bg = parse_color(bg, NYX_BG);
    msg.size = size && !strcmp(size, "small") ? LCD_TEXT_SIZE_SMALL : LCD_TEXT_SIZE_MEDIUM;
    msg.align = align && !strcmp(align, "right") ? LCD_TEXT_ALIGN_RIGHT :
        align && !strcmp(align, "center") ? LCD_TEXT_ALIGN_CENTER : LCD_TEXT_ALIGN_LEFT;
    esp_err_t err = lcd_renderer_navigate("home");
    dirty=true;
    bsp_display_unlock();
    return err;
}
void lcd_renderer_clear(void) {
    if (bsp_display_lock(2000) != ESP_OK) return;
    remove_image();
    count = 0;
    note[0] = 0;
    dirty=true;
    bsp_display_unlock();
}
static esp_err_t navigate_now(const char *target) {
    if (!phone || !target) return ESP_ERR_INVALID_ARG;
    if (strcmp(target, "home") && strcmp(target, "settings") && strcmp(target, "camera") &&
        strcmp(target, "scan") && strcmp(target, "info") && strcmp(target, "operator") &&
        strcmp(target, "wifi") && strcmp(target, "device_bind") && strcmp(target, "back")) return ESP_ERR_NOT_FOUND;
    if (bsp_display_lock(2000) != ESP_OK) return ESP_ERR_TIMEOUT;
    esp_err_t err = ESP_OK;
    using namespace esp_brookesia::apps;
    if (!strcmp(target, "back")) {
        err = phone->sendNavigateEvent(Manager::NavigateType::BACK) ? ESP_OK : ESP_FAIL;
    } else if (!strcmp(target, "settings") || !strcmp(target, "wifi") ||
               !strcmp(target, "info") || !strcmp(target, "operator") || !strcmp(target, "device_bind")) {
        auto settings = Settings::requestInstance();
        err = activate(settings);
        if (err == ESP_OK) {
            if (!strcmp(target, "wifi")) settings->showWifi();
            else if (!strcmp(target, "device_bind")) settings->showNyx();
            else if (!strcmp(target, "info") || !strcmp(target, "operator")) settings->showDevice(target);
            else settings->showRoot();
        }
    } else if (!strcmp(target, "camera") || !strcmp(target, "scan")) {
        const char *return_target = "settings";
        auto current = phone->getManager().getActiveApp();
        if (current && !strcmp(current->getName(), "Settings"))
            return_target = Settings::requestInstance()->remotePage();
        auto camera = Camera::requestInstance();
        camera->setScanMode(!strcmp(target, "scan"), return_target);
        err = activate(camera);
    } else {
        err = activate(app);
        app->render();
    }
    bsp_display_unlock();
    return err;
}
esp_err_t lcd_renderer_navigate(const char *target) {
    if (!phone || !target) return ESP_ERR_INVALID_ARG;
    const char *known[]={"home","settings","wifi","camera","scan","info","operator","device_bind","back"};
    bool valid=false;
    for (auto name:known) if (!strcmp(name,target)) valid=true;
    if (!valid) return ESP_ERR_NOT_FOUND;
    if (bsp_display_lock(1000)!=ESP_OK) return ESP_ERR_TIMEOUT;
    esp_err_t err=ESP_ERR_NO_MEM;
    if (navigation_count<4) {
        snprintf(pending_navigation[navigation_count++],24,"%s",target);
        err=ESP_OK;
    }
    bsp_display_unlock();
    return err;
}
esp_err_t lcd_renderer_show_device_bind(void) { return lcd_renderer_navigate("device_bind"); }
esp_err_t lcd_renderer_show_unregistered_mac(const char *, const char *) { return lcd_renderer_show_device_bind(); }
esp_err_t lcd_renderer_current_screen(const char **target, size_t *depth) {
    if (!phone || !target || !depth) return ESP_ERR_INVALID_ARG;
    if (bsp_display_lock(1000) != ESP_OK) return ESP_ERR_TIMEOUT;
    auto active = phone->getManager().getActiveApp();
    if (active == app) *target = "home";
    else if (active && !strcmp(active->getName(), "Settings")) *target = esp_brookesia::apps::Settings::requestInstance()->remotePage();
    else if (active && !strcmp(active->getName(), "Camera"))
        *target = esp_brookesia::apps::Camera::requestInstance()->scanning() ? "scan" : "camera";
    else *target = active ? "native_app" : "launcher";
    *depth = active ? 2 : 1;
    bsp_display_unlock();
    return ESP_OK;
}
}

static const char *TAG = "panel.gui";
extern "C" {
esp_err_t lcd_renderer_snapshot_jpeg(uint8_t *out, size_t cap, size_t *out_len)
{
    if (!out || !out_len || cap == 0u) return ESP_ERR_INVALID_ARG;
    *out_len = 0u;

    if (!screenshot_lock || xSemaphoreTake(screenshot_lock, 0) != pdTRUE) {
        return ESP_ERR_INVALID_STATE;
    }

    jpeg_encoder_handle_t encoder = nullptr;
    jpeg_encode_engine_cfg_t engine = {};
    engine.timeout_ms = 5000;
    esp_err_t enc_ready = jpeg_new_encoder_engine(&engine, &encoder);
    if (enc_ready != ESP_OK) {
        xSemaphoreGive(screenshot_lock);
        return enc_ready;
    }

    struct EncoderGuard { jpeg_encoder_handle_t h; ~EncoderGuard() { jpeg_del_encoder_engine(h); } } guard{encoder};

    if (bsp_display_lock(2000) != ESP_OK) {
        xSemaphoreGive(screenshot_lock);
        return ESP_ERR_TIMEOUT;
    }

    lv_display_t *display = lv_display_get_default();
    const void *scanout = nullptr;
    size_t scanout_size = 0;
    esp_err_t capture_err = panel_dpi_get_scanout_buffer(bsp_display_get_panel_handle(),
                                                         &scanout, &scanout_size);
    if (!display || capture_err != ESP_OK) {
        bsp_display_unlock();
        xSemaphoreGive(screenshot_lock);
        return capture_err == ESP_OK ? ESP_ERR_INVALID_STATE : capture_err;
    }
    int32_t w = lv_display_get_horizontal_resolution(display);
    int32_t h = lv_display_get_vertical_resolution(display);
    if (w <= 0 || h <= 0 || scanout_size != (size_t)w * h * 2u) {
        bsp_display_unlock();
        xSemaphoreGive(screenshot_lock);
        return ESP_ERR_INVALID_SIZE;
    }

    size_t raw_size = (size_t)w * (size_t)h * 2u;  // RGB565
    uint8_t *raw = (uint8_t *)heap_caps_aligned_alloc(64, raw_size,
                                           MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!raw) {
        bsp_display_unlock();
        ESP_LOGE(TAG, "snapshot raw alloc failed (%u bytes)", (unsigned)raw_size);
        xSemaphoreGive(screenshot_lock);
        return ESP_ERR_NO_MEM;
    }

    // Copy the actual DMA-selected frame, including system overlays. Redrawing
    // the object tree here can hide corruption in the frame the user sees.
    // Rendering is quiescent under the GUI lock; DMA only reads this buffer.
    capture_err = esp_cache_msync(const_cast<void *>(scanout), scanout_size,
                                  ESP_CACHE_MSYNC_FLAG_DIR_M2C);
    if (capture_err == ESP_OK) memcpy(raw, scanout, raw_size);
    bsp_display_unlock();
    if (capture_err != ESP_OK) {
        free(raw);
        ESP_LOGE(TAG, "scanout buffer read failed: %s", esp_err_to_name(capture_err));
        xSemaphoreGive(screenshot_lock);
        return capture_err;
    }

    // The HW encoder requires its output buffer to be allocated via
    // jpeg_alloc_encoder_mem (cache-line aligned, DMA-capable). We allocate a
    // private buffer the size of `cap`, encode into it, then memcpy into the
    // caller's buffer. Saves callers from having to know the alignment rules.
    jpeg_encode_memory_alloc_cfg_t mem_cfg = { .buffer_direction = JPEG_ENC_ALLOC_OUTPUT_BUFFER };
    size_t alloc_size = 0u;
    uint8_t *enc_out = (uint8_t *)jpeg_alloc_encoder_mem(cap, &mem_cfg, &alloc_size);
    if (!enc_out) {
        free(raw);
        ESP_LOGE(TAG, "jpeg_alloc_encoder_mem failed (%u bytes)", (unsigned)cap);
        xSemaphoreGive(screenshot_lock);
        return ESP_ERR_NO_MEM;
    }

    jpeg_encode_cfg_t enc_cfg = {
        .height       = (uint32_t)h,
        .width        = (uint32_t)w,
        .src_type     = JPEG_ENCODE_IN_FORMAT_RGB565,
        .sub_sample   = JPEG_DOWN_SAMPLING_YUV420,
        // Quality 40 keeps /lcd/snapshot.jpg below ~80 KiB on the typical
        // AI terminal scene. The HTTP response goes back through the
        // SDIO-backed WiFi link; bigger outbound frames trigger upstream
        // ESP-Hosted bug #144 (sdio_rx_get_buffer assert) and stall the
        // board. Fine quality for debug + HA mqtt.image / uitests use.
        .image_quality = 40,
    };
    uint32_t produced = 0u;
    // jpeg_alloc_encoder_mem calloc leaves dirty zero cache lines. Hand the
    // output to DMA before encoding so a later eviction cannot overwrite JPEG
    // bytes. The driver invalidates the payload again after DMA completes.
    esp_err_t err = esp_cache_msync(enc_out, alloc_size,
        ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_INVALIDATE);
    if (err == ESP_OK)
        err = jpeg_encoder_process(encoder, &enc_cfg, raw, raw_size,
                                    enc_out, (uint32_t)alloc_size, &produced);
    free(raw);
    if (err != ESP_OK) {
        free(enc_out);
        ESP_LOGE(TAG, "jpeg_encoder_process: %s", esp_err_to_name(err));
        xSemaphoreGive(screenshot_lock);
        return err;
    }
    if ((size_t)produced > cap) {
        free(enc_out);
        ESP_LOGE(TAG, "jpeg too large for caller buf: %u > %u",
                 (unsigned)produced, (unsigned)cap);
        xSemaphoreGive(screenshot_lock);
        return ESP_ERR_NO_MEM;
    }
    memcpy(out, enc_out, produced);
    free(enc_out);
    *out_len = produced;
    ESP_LOGI(TAG, "lcd snapshot %ldx%ld -> %u bytes JPEG",
             (long)w, (long)h, (unsigned)produced);
    xSemaphoreGive(screenshot_lock);
    return ESP_OK;
}

}
