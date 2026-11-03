#pragma once
#include <stdbool.h>
#include <stddef.h>

// The caller serializes task/ISR access. A periodic interrupt or completion
// selecting the old framebuffer cannot release the submitted frame's wait.
typedef struct {
    const void *submitted;
    bool pending;
} panel_scanout_gate_t;

static inline void panel_scanout_submit(panel_scanout_gate_t *gate, const void *frame)
{
    gate->submitted = frame;
    gate->pending = true;
}

static inline bool panel_scanout_complete(panel_scanout_gate_t *gate, const void *active)
{
    if (!gate->pending || !active || gate->submitted != active) return false;
    gate->pending = false;
    return true;
}
