#pragma once

#include "net_manager.h"

#include <stdbool.h>

bool provisioning_mode_active(void);
void provisioning_mode_enter(const char *reason);
void provisioning_mode_exit(void);
void provisioning_mode_note_wifi_failure(const net_manager_snapshot_t *snapshot);
void provisioning_mode_note_network_up(void);
