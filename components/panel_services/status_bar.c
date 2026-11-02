#include "status_bar.h"

static const char *net_symbol(bool link, bool ip)
{
    if (ip) {
        return "●";
    }
    if (link) {
        return "◐";
    }
    return "○";
}

static const char *nyx_symbol(nyx_node_state_t state)
{
    switch (state) {
    case NYX_NODE_UP:
        return "●";
    case NYX_NODE_CONNECTING:
        return "◐";
    case NYX_NODE_DOWN:
    default:
        return "○";
    }
}

void status_bar_format(const net_manager_snapshot_t *net,
                       nyx_node_state_t nyx_state,
                       const char *slug,
                       status_bar_text_t *out)
{
    if (!out) {
        return;
    }
    static const net_manager_snapshot_t empty = {0};
    if (!net) {
        net = &empty;
    }

    out->eth = net_symbol(net->eth_link, net->eth_ip);
    out->wifi = net_symbol(net->wifi_link, net->wifi_ip);
    out->nyx = nyx_symbol(nyx_state);
    out->footer = (slug && slug[0]) ? slug : "nyx: unregistered";
}

