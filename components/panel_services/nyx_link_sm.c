#include "nyx_link_sm.h"

#define NYX_LINK_BACKOFF_BASE_MS 1000u
#define NYX_LINK_BACKOFF_MAX_MS  60000u

static uint32_t calc_backoff(uint8_t failures)
{
    uint32_t ms = NYX_LINK_BACKOFF_BASE_MS;
    for (uint8_t i = 1; i < failures; i++) {
        if (ms >= NYX_LINK_BACKOFF_MAX_MS / 2u) {
            return NYX_LINK_BACKOFF_MAX_MS;
        }
        ms *= 2u;
    }
    return ms > NYX_LINK_BACKOFF_MAX_MS ? NYX_LINK_BACKOFF_MAX_MS : ms;
}

void nyx_link_sm_init(nyx_link_sm_t *sm)
{
    if (!sm) {
        return;
    }
    sm->state = NYX_LINK_DISCONNECTED;
    sm->failures = 0;
    sm->backoff_ms = 0;
}

void nyx_link_sm_apply(nyx_link_sm_t *sm, nyx_link_evt_t evt)
{
    if (!sm) {
        return;
    }

    if (evt == NYX_LINK_EVT_NETIF_DOWN) {
        sm->state = NYX_LINK_DISCONNECTED;
        sm->backoff_ms = 0;
        return;
    }

    switch (sm->state) {
    case NYX_LINK_DISCONNECTED:
        if (evt == NYX_LINK_EVT_NETIF_UP) {
            sm->state = NYX_LINK_CONNECTING;
            sm->backoff_ms = 0;
        } else if (evt == NYX_LINK_EVT_WSS_ERR) {
            sm->state = NYX_LINK_RECONNECTING;
            sm->failures++;
            sm->backoff_ms = calc_backoff(sm->failures);
        }
        break;
    case NYX_LINK_CONNECTING:
        if (evt == NYX_LINK_EVT_WSS_ACK) {
            sm->state = NYX_LINK_CONNECTED;
            sm->failures = 0;
            sm->backoff_ms = 0;
        } else if (evt == NYX_LINK_EVT_WSS_ERR) {
            sm->state = NYX_LINK_RECONNECTING;
            sm->failures++;
            sm->backoff_ms = calc_backoff(sm->failures);
        }
        break;
    case NYX_LINK_CONNECTED:
        if (evt == NYX_LINK_EVT_WSS_ERR || evt == NYX_LINK_EVT_IFACE_CHANGED) {
            sm->state = NYX_LINK_RECONNECTING;
            sm->failures++;
            sm->backoff_ms = calc_backoff(sm->failures);
        }
        break;
    case NYX_LINK_RECONNECTING:
        if (evt == NYX_LINK_EVT_RETRY_TIMER || evt == NYX_LINK_EVT_NETIF_UP) {
            sm->state = NYX_LINK_CONNECTING;
            sm->backoff_ms = 0;
        } else if (evt == NYX_LINK_EVT_WSS_ERR) {
            sm->failures++;
            sm->backoff_ms = calc_backoff(sm->failures);
        } else if (evt == NYX_LINK_EVT_WSS_ACK) {
            sm->state = NYX_LINK_CONNECTED;
            sm->failures = 0;
            sm->backoff_ms = 0;
        }
        break;
    }
}

nyx_link_state_t nyx_link_sm_state(const nyx_link_sm_t *sm)
{
    return sm ? sm->state : NYX_LINK_DISCONNECTED;
}

uint32_t nyx_link_sm_backoff_ms(const nyx_link_sm_t *sm)
{
    return sm ? sm->backoff_ms : 0;
}
