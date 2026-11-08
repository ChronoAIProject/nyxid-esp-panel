#pragma once
#include "esp_err.h"
#include "panel_update_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool automatic, checking, available;
    char latest_version[32], blocked_version[32], state[32];
    esp_err_t last_error;
} panel_update_snapshot_t;

void panel_update_init(void);
void panel_update_snapshot(panel_update_snapshot_t *out);
const char *panel_update_profile(void);
esp_err_t panel_update_set_automatic(bool enabled);
// Check-only or check-and-install. Always asynchronous, install waits for idle.
esp_err_t panel_update_check(bool install);
// Persist before selecting a boot slot; a rolled-back version is not retried automatically.
esp_err_t panel_update_record_pending(const char *version);

#ifdef __cplusplus
}
#endif
