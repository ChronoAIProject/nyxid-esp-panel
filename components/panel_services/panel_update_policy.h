#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PANEL_RELEASE_PROJECT "nyxid-esp-panel"
#define PANEL_RELEASE_ROOT "https://github.com/ChronoAIProject/nyxid-esp-panel/releases/"
#define PANEL_RELEASE_SLOT_SIZE 0x690000u

typedef struct {
    char version[32];
    char url[256];
    uint8_t sha256[32];
    size_t size;
} panel_release_t;

bool panel_update_version_valid(const char *version);
bool panel_update_version_newer(const char *candidate, const char *running);
bool panel_update_download_url_allowed(const char *url);
// Parse only stable releases for this exact hardware/build profile.
bool panel_update_parse_release(const void *json, size_t length, const char *profile,
                                panel_release_t *out);
