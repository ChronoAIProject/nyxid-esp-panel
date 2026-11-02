#include "nyx_upload.h"
#include <string.h>
static int unhex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
void nyx_upload_cancel(nyx_upload_t *u) { if (u) { u->active = false; u->used = 0; } }
bool nyx_upload_open(nyx_upload_t *u, const char *uuid, uint8_t *buffer,
                     size_t cap, bool have_expected, size_t expected) {
    if (!u || u->active || !uuid || strlen(uuid) != 36 || !buffer || !cap ||
        cap > NYX_UPLOAD_MAX_BYTES || (have_expected && expected > cap)) return false;
    uint8_t id[16];
    size_t n = 0;
    for (size_t i = 0; i < 36;) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (uuid[i++] != '-') return false;
            continue;
        }
        int a = unhex(uuid[i++]), b = unhex(uuid[i++]);
        if (a < 0 || b < 0) return false;
        id[n++] = (uint8_t)((a << 4) | b);
    }
    memset(u, 0, sizeof(*u));
    memcpy(u->id, id, 16);
    u->buffer = buffer; u->cap = cap; u->expected = expected;
    u->have_expected = have_expected; u->active = true;
    return true;
}
nyx_upload_result_t nyx_upload_frame(nyx_upload_t *u, const uint8_t *data, size_t size) {
    if (!u || !u->active) return NYX_UPLOAD_INVALID;
    if (!data || size < NYX_UPLOAD_HEADER_BYTES || memcmp(data, "NYXM", 4) ||
        (data[4] != 8 && data[4] != 9) || data[5] > 1 || memcmp(data + 6, u->id, 16)) goto invalid;
    if (data[4] == 9) { nyx_upload_cancel(u); return NYX_UPLOAD_ABORTED; }
    uint64_t sequence = 0;
    for (int i = 22; i < 30; ++i) sequence = (sequence << 8) | data[i];
    size_t bytes = size - NYX_UPLOAD_HEADER_BYTES;
    if (sequence != u->sequence || bytes > u->cap - u->used ||
        (u->have_expected && bytes > u->expected - u->used)) goto invalid;
    memcpy(u->buffer + u->used, data + NYX_UPLOAD_HEADER_BYTES, bytes);
    u->used += bytes;
    ++u->sequence;
    if (!data[5]) return NYX_UPLOAD_MORE;
    if (u->have_expected && u->used != u->expected) goto invalid;
    u->active = false;
    return NYX_UPLOAD_DONE;
invalid:
    nyx_upload_cancel(u);
    return NYX_UPLOAD_INVALID;
}
