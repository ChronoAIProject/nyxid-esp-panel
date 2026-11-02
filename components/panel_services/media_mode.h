#pragma once

#include "esp_err.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MEDIA_MODE_PUBLISH = 0,
    MEDIA_MODE_PREVIEW,
} media_mode_t;

void media_mode_init(void);
esp_err_t media_mode_switch(media_mode_t mode, const char *reason);
esp_err_t media_mode_request_publish(const char *reason);
esp_err_t media_mode_request_preview(const char *reason);
media_mode_t media_mode_current(void);
const char *media_mode_name(media_mode_t mode);
bool media_mode_is_preview(void);

#ifdef __cplusplus
}
#endif
