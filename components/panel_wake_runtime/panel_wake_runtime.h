#pragma once
#include "esp_wn_iface.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental detector adapter only. The application retains its existing
// single I2S/AEC/AGC producer, audio ownership and ordered call FIFO.
extern const esp_wn_iface_t panel_wake_runtime_iface;
int panel_wake_runtime_prime_samples(void);

#ifdef __cplusplus
}
#endif
