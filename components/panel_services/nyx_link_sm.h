#pragma once

#include <stdint.h>

typedef enum {
    NYX_LINK_DISCONNECTED = 0,
    NYX_LINK_CONNECTING,
    NYX_LINK_CONNECTED,
    NYX_LINK_RECONNECTING,
} nyx_link_state_t;

typedef enum {
    NYX_LINK_EVT_NETIF_UP = 0,
    NYX_LINK_EVT_NETIF_DOWN,
    NYX_LINK_EVT_WSS_ACK,
    NYX_LINK_EVT_WSS_ERR,
    NYX_LINK_EVT_IFACE_CHANGED,
    NYX_LINK_EVT_RETRY_TIMER,
} nyx_link_evt_t;

typedef struct {
    nyx_link_state_t state;
    uint8_t failures;
    uint32_t backoff_ms;
} nyx_link_sm_t;

void nyx_link_sm_init(nyx_link_sm_t *sm);
void nyx_link_sm_apply(nyx_link_sm_t *sm, nyx_link_evt_t evt);
nyx_link_state_t nyx_link_sm_state(const nyx_link_sm_t *sm);
uint32_t nyx_link_sm_backoff_ms(const nyx_link_sm_t *sm);
