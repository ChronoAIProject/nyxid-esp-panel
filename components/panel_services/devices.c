#include "devices.h"
#include "esp_mac.h"
#include <stdio.h>

static char mac_text[18];
static char label[40];
static device_config_t identity;

const device_config_t *devices_find_self(void)
{
    if (!mac_text[0]) {
        uint8_t mac[6] = {0};
        if (esp_efuse_mac_get_default(mac) != ESP_OK) return NULL;
        snprintf(mac_text, sizeof(mac_text), "%02x:%02x:%02x:%02x:%02x:%02x",
                 mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        snprintf(label, sizeof(label), "nyx-panel-%02x%02x%02x", mac[3], mac[4], mac[5]);
        identity.mac = mac_text;
        identity.label = label;
        identity.site = "";
    }
    return &identity;
}

const device_config_t *devices_self(void) { return devices_find_self(); }
const char *devices_self_mac_str(void)
{
    devices_find_self();
    return mac_text;
}
