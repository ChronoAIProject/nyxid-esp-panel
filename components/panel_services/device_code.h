#pragma once
#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DEVICE_LOGIN_IDLE,
    DEVICE_LOGIN_REQUESTING,
    DEVICE_LOGIN_WAITING,
    DEVICE_LOGIN_REGISTERING,
    DEVICE_LOGIN_SERVICE,
    DEVICE_LOGIN_READY,
    DEVICE_LOGIN_CANCELLED,
    DEVICE_LOGIN_FAILED,
} device_login_state_t;

typedef struct {
    device_login_state_t state;
    char user_code[16];
    char verification_uri[192];
    char verification_uri_complete[384];
    char message[160];
    uint32_t seconds_remaining;
    bool busy;
    bool node_setup;
} device_login_snapshot_t;

// Explicitly initiated from Settings. Never starts automatically on an upgrade.
esp_err_t device_code_start_if_needed(void);
// Separate first-party setup for a new remote-control node.
esp_err_t device_code_start_node_setup(void);
void device_code_cancel(void);
void device_code_snapshot(device_login_snapshot_t *snapshot);

#ifdef __cplusplus
}
#endif
