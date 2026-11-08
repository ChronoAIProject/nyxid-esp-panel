#pragma once
#include <stdbool.h>
#include "esp_err.h"
// Shared by URL OTA and the LAN streaming image endpoint.
bool panel_ota_try_lock(void);
void panel_ota_unlock(void);
bool panel_ota_busy(void);
// Stage names must have static storage duration.
void panel_ota_progress(const char *stage, esp_err_t error);
const char *panel_ota_stage(void);
esp_err_t panel_ota_error(void);

#include <stddef.h>
#include <stdint.h>
// Optional release version/size enables strict release identity and HTTPS checks.
esp_err_t panel_ota_start_url(const char *url, const uint8_t expected[32],
                              const char *version, size_t size);
