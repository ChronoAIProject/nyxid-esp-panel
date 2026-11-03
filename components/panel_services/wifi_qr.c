#include "wifi_qr.h"
#include <string.h>

bool wifi_qr_parse(const char *payload, wifi_qr_t *out)
{
    if (!payload || !out) return false;
    memset(out, 0, sizeof(*out));
    if (strncmp(payload, "WIFI:", 5)) return false;
    const char *cursor = payload + 5;
    char type[16] = "";
    unsigned seen = 0;
    while (*cursor) {
        if (*cursor == ';') { ++cursor; continue; }
        const char *key = cursor;
        while (*cursor && *cursor != ':' && *cursor != ';') ++cursor;
        if (*cursor != ':' || cursor - key != 1) return false;
        ++cursor;
        char value[128];
        size_t length = 0;
        while (*cursor && *cursor != ';') {
            if (*cursor == '\\') {
                ++cursor;
                if (!*cursor || !strchr("\\;:,\"", *cursor)) return false;
            }
            if (length + 1 >= sizeof(value)) return false;
            value[length++] = *cursor++;
        }
        if (*cursor != ';') return false;
        ++cursor;
        value[length] = 0;
        unsigned bit;
        char *destination;
        size_t cap;
        switch (*key) {
        case 'S': bit = 1; destination = out->ssid; cap = sizeof(out->ssid); break;
        case 'P': bit = 2; destination = out->password; cap = sizeof(out->password); break;
        case 'T': bit = 4; destination = type; cap = sizeof(type); break;
        case 'H':
            if (seen & 8 || (strcmp(value, "true") && strcmp(value, "false"))) return false;
            seen |= 8;
            out->hidden = !strcmp(value, "true");
            continue;
        default: return false; // Enterprise/EAP and vendor fields need a separate UI.
        }
        if (seen & bit || length >= cap) return false;
        seen |= bit;
        memcpy(destination, value, length + 1);
    }
    if (!(seen & 1) || !out->ssid[0]) return false;
    if (!strcmp(type, "nopass") || !type[0]) return !out->password[0];
    // C6 supports personal WPA/WPA2/WPA3 through the shared station driver.
    if (strcmp(type, "WPA") && strcmp(type, "WPA2") && strcmp(type, "WPA3")) return false;
    return strlen(out->password) >= 8;
}
