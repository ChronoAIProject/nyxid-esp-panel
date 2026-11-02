#include "nyx_ws_rx.h"
#include <stdlib.h>
#include <string.h>
void nyx_ws_rx_reset(nyx_ws_rx_t *rx) {
    if (rx) { free(rx->data); memset(rx,0,sizeof(*rx)); }
}
int nyx_ws_rx_feed(nyx_ws_rx_t *rx, uint8_t opcode, bool fin, size_t total,
                   size_t offset, const void *chunk, size_t len,
                   uint8_t **out, size_t *out_size, uint8_t *out_opcode) {
    if (!rx || !out || !out_size || !out_opcode) return -1;
    *out = NULL;
    if ((opcode != 0 && opcode != 1 && opcode != 2) || (len && !chunk) ||
        total > NYX_WS_MAX_MESSAGE || offset > total || len > total-offset) goto invalid;
    if (!offset) {
        if (rx->frame_active) goto invalid;
        if (opcode == 1 || opcode == 2) {
            if (rx->message_active) goto invalid;
            rx->message_active = true; rx->opcode = opcode;
        } else if (!rx->message_active) goto invalid;
        if (total > NYX_WS_MAX_MESSAGE-rx->used) goto invalid;
        uint8_t *buffer = realloc(rx->data,rx->used+total+1);
        if (!buffer) goto invalid;
        rx->data = buffer;
        rx->frame_start=rx->used; rx->frame_size=total; rx->frame_used=0; rx->frame_active=true;
    }
    if (!rx->frame_active || total != rx->frame_size || offset != rx->frame_used) goto invalid;
    if (len) memcpy(rx->data+rx->frame_start+offset,chunk,len);
    rx->frame_used+=len;
    if (rx->frame_used < total) return 0;
    rx->used+=total; rx->frame_active=false;
    if (!fin) return 0;
    rx->data[rx->used]=0;
    *out=rx->data; *out_size=rx->used; *out_opcode=rx->opcode;
    memset(rx,0,sizeof(*rx));
    return 1;
invalid:
    nyx_ws_rx_reset(rx); return -1;
}
