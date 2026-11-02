#pragma once
#include "esp_err.h"
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
esp_err_t panel_runtime_start(void);
bool panel_runtime_ready(void);
void panel_runtime_set_preview(bool enabled);
bool panel_runtime_preview_ready(void);
esp_err_t panel_runtime_preview_error(void);
#ifdef __cplusplus
}
#endif
