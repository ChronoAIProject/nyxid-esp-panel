#pragma once

#include "esp_err.h"

#include <stdbool.h>

typedef struct {
    char state[16];
    char host[96];
    char last_error[96];
    bool enabled;
    bool started;
    bool connected;
} mqtt_bridge_status_t;

esp_err_t mqtt_bridge_start(void);
void mqtt_bridge_stop(void);
void mqtt_bridge_reload(void);
bool mqtt_bridge_is_started(void);
bool mqtt_bridge_is_connected(void);
void mqtt_bridge_get_status(mqtt_bridge_status_t *status);
