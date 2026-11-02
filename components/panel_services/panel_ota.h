#pragma once
#include <stdbool.h>
#include "esp_err.h"
// Shared by URL OTA and the LAN streaming image endpoint.
bool panel_ota_try_lock(void);
void panel_ota_unlock(void);
