#pragma once
#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { MEDIA_MODE_IDLE = 0, MEDIA_MODE_PREVIEW, MEDIA_MODE_CAPTURE, MEDIA_MODE_ERROR } media_mode_t;
typedef struct {
    media_mode_t mode;
    uint32_t completed, failed, last_capture_ms;
    esp_err_t last_error;
} camera_status_t;
void media_mode_init(void);
esp_err_t media_mode_request_idle(const char *reason);
esp_err_t media_mode_request_preview(const char *reason);
media_mode_t media_mode_current(void);
const char *media_mode_name(media_mode_t mode);
bool media_mode_is_preview(void);
void media_mode_status(camera_status_t *status);
// allow_start=false limits LAN to an already visible native preview.
// Concurrent requests fail busy; the returned buffer belongs to the caller.
esp_err_t media_mode_snapshot(uint8_t *dst, size_t cap, size_t *out_len, bool allow_start);
#ifdef __cplusplus
}
#endif
