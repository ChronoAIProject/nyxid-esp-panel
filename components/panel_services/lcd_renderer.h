#pragma once

#include "net_manager.h"

#include "esp_err.h"

#include <stddef.h>
#include <stdint.h>

esp_err_t lcd_renderer_init(void);
esp_err_t lcd_renderer_start(void);
void lcd_renderer_set_net(const net_manager_snapshot_t *snapshot);
void lcd_renderer_refresh_status(void);
void lcd_renderer_reload_config(void);
void lcd_renderer_set_message(const char *text);
void lcd_renderer_note_qr_decoded(const char *payload_type);
esp_err_t lcd_renderer_show_device_bind(void);
esp_err_t lcd_renderer_show_unregistered_mac(const char *mac, const char *reason);
esp_err_t lcd_renderer_push_image(const uint8_t *jpeg, size_t jpeg_len, uint32_t duration_ms);
esp_err_t lcd_renderer_push_text(const char *text, const char *size, const char *color,
                                 const char *bg, const char *align, uint32_t duration_ms);
void lcd_renderer_clear(void);

// Capture the current LCD framebuffer and JPEG-encode it into out.
// out_len is set to the produced JPEG size on success.
// Designed for debug / remote-screenshot scenarios — uses the HW JPEG encoder.
esp_err_t lcd_renderer_snapshot_jpeg(uint8_t *out, size_t cap, size_t *out_len);

// Remote screen navigation. target is one of: "home", "settings", "scan",
// "camera", "info", "operator", "wifi", "device_bind", "back".
// Settings owns network, status and NyxID setup pages; Camera owns scanning.
// "back" invokes native navigation (scan returns to its originating Settings page).
esp_err_t lcd_renderer_navigate(const char *target);

// Read-only: which screen is currently on top + how deep is the stack.
// out_target points to a static string ("home"/"settings"/etc) — do not free.
esp_err_t lcd_renderer_current_screen(const char **out_target, size_t *out_depth);
