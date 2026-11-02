#include "stream_sm.h"

static bool can_start(const stream_sm_t *sm)
{
    return sm->net_up && sm->target_set;
}

void stream_sm_init(stream_sm_t *sm)
{
    if (!sm) {
        return;
    }
    sm->state = STREAM_IDLE;
    sm->net_up = false;
    sm->target_set = false;
    sm->retry_pending = false;
}

void stream_sm_apply(stream_sm_t *sm, stream_evt_t evt)
{
    if (!sm) {
        return;
    }

    switch (evt) {
    case STREAM_EVT_NET_UP:
        sm->net_up = true;
        if (sm->state == STREAM_FAILED) {
            sm->state = STREAM_IDLE;
            sm->retry_pending = false;
        }
        break;
    case STREAM_EVT_NET_DOWN:
        sm->net_up = false;
        sm->retry_pending = false;
        if (sm->state == STREAM_STARTING || sm->state == STREAM_PUBLISHING || sm->state == STREAM_FAILED) {
            sm->state = STREAM_FAILED;
        }
        break;
    case STREAM_EVT_TARGET_SET:
        sm->target_set = true;
        break;
    case STREAM_EVT_TARGET_CLEAR:
        sm->target_set = false;
        sm->retry_pending = false;
        sm->state = STREAM_IDLE;
        break;
    case STREAM_EVT_START:
        if (sm->state == STREAM_IDLE && can_start(sm)) {
            sm->state = STREAM_STARTING;
            sm->retry_pending = false;
        }
        break;
    case STREAM_EVT_STOP:
        sm->state = STREAM_IDLE;
        sm->retry_pending = false;
        break;
    case STREAM_EVT_WHIP_OK:
        if (sm->state == STREAM_STARTING) {
            sm->state = STREAM_PUBLISHING;
            sm->retry_pending = false;
        }
        break;
    case STREAM_EVT_WHIP_ERR:
        sm->state = STREAM_FAILED;
        sm->retry_pending = sm->net_up && sm->target_set;
        break;
    case STREAM_EVT_RETRY_TIMER:
        if (sm->state == STREAM_FAILED && sm->retry_pending && can_start(sm)) {
            sm->state = STREAM_STARTING;
            sm->retry_pending = false;
        }
        break;
    }
}

stream_state_t stream_sm_state(const stream_sm_t *sm)
{
    return sm ? sm->state : STREAM_IDLE;
}

bool stream_sm_should_retry(const stream_sm_t *sm)
{
    return sm && sm->state == STREAM_FAILED && sm->retry_pending;
}
