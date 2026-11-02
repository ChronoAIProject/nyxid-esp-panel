#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t panel_flash_init(void);
// Also protect image verification, which reads through an MMU mapping rather
// than esp_flash_read. Calls nest with the wrapped flash APIs below.
void panel_flash_lock(void);
void panel_flash_unlock(void);

#ifdef __cplusplus
}
#endif
