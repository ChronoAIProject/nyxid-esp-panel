#pragma once
#include <stdbool.h>

typedef struct {
    char ssid[33];
    char password[64];
    bool hidden;
} wifi_qr_t;

// Parses a standard WIFI: code. No side effects and no silent truncation.
bool wifi_qr_parse(const char *payload, wifi_qr_t *out);
