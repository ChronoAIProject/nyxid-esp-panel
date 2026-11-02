#pragma once

#include <stdbool.h>

typedef enum {
    STREAM_IDLE = 0,
    STREAM_STARTING,
    STREAM_PUBLISHING,
    STREAM_FAILED,
} stream_state_t;

typedef enum {
    STREAM_EVT_NET_UP = 0,
    STREAM_EVT_NET_DOWN,
    STREAM_EVT_TARGET_SET,
    STREAM_EVT_TARGET_CLEAR,
    STREAM_EVT_START,
    STREAM_EVT_STOP,
    STREAM_EVT_WHIP_OK,
    STREAM_EVT_WHIP_ERR,
    STREAM_EVT_RETRY_TIMER,
} stream_evt_t;

typedef struct {
    stream_state_t state;
    bool net_up;
    bool target_set;
    bool retry_pending;
} stream_sm_t;

void stream_sm_init(stream_sm_t *sm);
void stream_sm_apply(stream_sm_t *sm, stream_evt_t evt);
stream_state_t stream_sm_state(const stream_sm_t *sm);
bool stream_sm_should_retry(const stream_sm_t *sm);
