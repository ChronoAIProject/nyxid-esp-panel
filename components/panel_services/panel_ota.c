#include "panel_ota.h"
#include <stdatomic.h>
static atomic_flag busy = ATOMIC_FLAG_INIT;
bool panel_ota_try_lock(void) { return !atomic_flag_test_and_set(&busy); }
void panel_ota_unlock(void) { atomic_flag_clear(&busy); }
