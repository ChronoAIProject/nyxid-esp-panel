#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    NET_IFACE_WIFI = 0,
    NET_IFACE_ETH,
} net_iface_t;

typedef struct {
    bool wifi_link;
    bool wifi_ip;
    bool eth_link;
    bool eth_ip;
    char wifi_ip_addr[16];
    char eth_ip_addr[16];
} net_manager_snapshot_t;

typedef void (*net_manager_cb_t)(const net_manager_snapshot_t *snapshot, void *user);

void net_manager_init(void);
void net_manager_subscribe(net_manager_cb_t cb, void *user);
void net_manager_note_legacy_network(bool connected);
net_manager_snapshot_t net_manager_snapshot(void);
bool net_manager_connected(void);
