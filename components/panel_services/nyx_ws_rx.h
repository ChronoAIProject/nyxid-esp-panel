#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define NYX_WS_MAX_MESSAGE (96u * 1024u)
typedef struct {
    uint8_t *data;
    size_t used, frame_start, frame_size, frame_used;
    uint8_t opcode;
    bool message_active, frame_active;
} nyx_ws_rx_t;
void nyx_ws_rx_reset(nyx_ws_rx_t *rx);
// Takes ownership of completed data in *out (NUL terminated, even for binary).
// -1 invalid / allocation failure, 0 partial, 1 complete.
int nyx_ws_rx_feed(nyx_ws_rx_t *rx, uint8_t opcode, bool fin, size_t total,
                   size_t offset, const void *chunk, size_t len,
                   uint8_t **out, size_t *out_size, uint8_t *out_opcode);
