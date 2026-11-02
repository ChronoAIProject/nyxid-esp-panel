#pragma once

#include "net_manager.h"
#include "nyx_node.h"

#include <stdbool.h>

typedef struct {
    const char *eth;
    const char *wifi;
    const char *nyx;
    const char *footer;
} status_bar_text_t;

void status_bar_format(const net_manager_snapshot_t *net,
                       nyx_node_state_t nyx_state,
                       const char *slug,
                       status_bar_text_t *out);

