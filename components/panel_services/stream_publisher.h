#pragma once

#include "esp_err.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define STREAM_PUBLISHER_TARGET_MAX 256u

typedef struct {
    char target_whip_url[STREAM_PUBLISHER_TARGET_MAX];
    bool auto_start;
    uint8_t fps;
    uint32_t bitrate;
} stream_publisher_config_t;

typedef struct {
    char state[16];
    char target_whip_url[STREAM_PUBLISHER_TARGET_MAX];
    bool auto_start;
    uint8_t fps;
    uint32_t bitrate;
    char last_error[96];
} stream_publisher_status_t;

void stream_publisher_init(void);
esp_err_t stream_publisher_set_config(const stream_publisher_config_t *cfg, bool persist);
esp_err_t stream_publisher_get_config(stream_publisher_config_t *cfg);
esp_err_t stream_publisher_set_target(const char *target_whip_url);
esp_err_t stream_publisher_start(void);
esp_err_t stream_publisher_stop(void);
void stream_publisher_suspend_for_preview(void);
void stream_publisher_resume_after_preview(void);
void stream_publisher_poll(void);
void stream_publisher_get_status(stream_publisher_status_t *status);
void stream_publisher_on_webrtc_connected(void);
void stream_publisher_on_webrtc_failed(const char *reason);
