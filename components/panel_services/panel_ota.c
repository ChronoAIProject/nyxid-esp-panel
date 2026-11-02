#include "panel_ota.h"
#include <stdatomic.h>
static atomic_bool busy;
static _Atomic(const char *) stage = "idle";
static atomic_int last_error;
bool panel_ota_try_lock(void)
{
    bool expected = false;
    return atomic_compare_exchange_strong(&busy, &expected, true);
}
void panel_ota_unlock(void) { atomic_store(&busy, false); }
bool panel_ota_busy(void) { return atomic_load(&busy); }
void panel_ota_progress(const char *value, esp_err_t error)
{
    atomic_store(&last_error, error);
    atomic_store(&stage, value);
}
const char *panel_ota_stage(void) { return atomic_load(&stage); }
esp_err_t panel_ota_error(void) { return atomic_load(&last_error); }
