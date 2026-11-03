#pragma once
#include "esp_lcd_mipi_dsi.h"

// ESP-IDF 5.5 has VSYNC but no public DMA frame-complete notification. This
// project-local extension reports the framebuffer actually selected by DMA,
// after the previous transfer has retired. Register/unregister while quiescent.
typedef bool (*panel_dpi_scanout_cb_t)(esp_lcd_panel_handle_t panel,
                                     const void *active_frame, void *ctx);
esp_err_t panel_dpi_register_scanout_callback(esp_lcd_panel_handle_t panel,
                                            panel_dpi_scanout_cb_t callback, void *ctx);
