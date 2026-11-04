#pragma once
#include <stdbool.h>
#include <stddef.h>
bool panel_management_slug(const char *daily, char *out, size_t cap);
// Called only for requests delivered by the authenticated NyxID node tunnel.
bool panel_service_can_call(const char *daily, const char *service_slug,
    const char *method, const char *path);
