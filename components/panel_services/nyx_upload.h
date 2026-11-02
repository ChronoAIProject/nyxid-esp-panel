#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define NYX_UPLOAD_MAX_BYTES 65536u
#define NYX_UPLOAD_HEADER_BYTES 30u

typedef struct {
    uint8_t id[16];
    uint64_t sequence;
    uint8_t *buffer;
    size_t cap, used, expected;
    bool active, have_expected;
} nyx_upload_t;
typedef enum { NYX_UPLOAD_MORE, NYX_UPLOAD_DONE, NYX_UPLOAD_ABORTED, NYX_UPLOAD_INVALID } nyx_upload_result_t;
// All calls belong to one consumer task; the caller owns buffer and metadata.
bool nyx_upload_open(nyx_upload_t *upload, const char *uuid, uint8_t *buffer,
                     size_t capacity, bool have_expected, size_t expected);
nyx_upload_result_t nyx_upload_frame(nyx_upload_t *upload, const uint8_t *data, size_t size);
void nyx_upload_cancel(nyx_upload_t *upload);
