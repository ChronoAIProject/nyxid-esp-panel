#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *mac;              // "aa:bb:cc:dd:ee:ff" lowercase, with colons
    const char *label;            // operator-facing
    const char *site;             // "home" | "office" - metadata only
    const char *slug;             // Nyx node slug; MQTT topic prefix haCam/<slug>/...
    const char *wifi_ssid_1;
    const char *wifi_password_1;
    const char *wifi_ssid_2;      // "" means no backup AP - net_manager skips
    const char *wifi_password_2;
    const char *ha_host;
    const char *mqtt_host;
    uint16_t    mqtt_port;
    const char *mqtt_user;
    const char *mqtt_pass;
    const char *nyx_api_key;
    const char *nyx_api_key_http;
    const char *nyx_node_id;
    const char *nyx_signing_secret;
} device_config_t;

extern const device_config_t DEVICE_CONFIGS[];
extern const size_t DEVICE_CONFIGS_COUNT;

// Reads MAC, scans DEVICE_CONFIGS, returns entry pointer or NULL.
// Result is cached on first successful call. Idempotent.
const device_config_t *devices_find_self(void);

// Returns cached pointer from last devices_find_self() (NULL if never called
// or last call returned NULL). Use this from defaults_* functions.
const device_config_t *devices_self(void);

// Returns the MAC string read at boot, "aa:bb:cc:dd:ee:ff", in a static buffer.
// Useful for the red-splash halt screen and for logging.
const char *devices_self_mac_str(void);

#ifdef __cplusplus
}
#endif
