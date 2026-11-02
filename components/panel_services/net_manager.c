#include "net_manager.h"

#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "provisioning.h"

#ifdef CONFIG_NETWORK_USE_ETHERNET
#include "esp_eth.h"
#endif

#include <string.h>

static const char *TAG = "haCam.net";

#define NET_MANAGER_MAX_SUBS 6

typedef struct {
    net_manager_cb_t cb;
    void *user;
} subscriber_t;

static struct {
    net_manager_snapshot_t snapshot;
    subscriber_t subs[NET_MANAGER_MAX_SUBS];
    SemaphoreHandle_t lock;
    bool initialized;
} g;

static void ensure_lock(void)
{
    if (!g.lock) {
        g.lock = xSemaphoreCreateMutex();
    }
}

static void notify_subscribers(const net_manager_snapshot_t *snapshot)
{
    subscriber_t subs[NET_MANAGER_MAX_SUBS];
    size_t count = 0;

    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    for (size_t i = 0; i < NET_MANAGER_MAX_SUBS; ++i) {
        if (g.subs[i].cb) {
            subs[count++] = g.subs[i];
        }
    }
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }

    for (size_t i = 0; i < count; ++i) {
        subs[i].cb(snapshot, subs[i].user);
    }
}

static void update_snapshot(const net_manager_snapshot_t *next)
{
    net_manager_snapshot_t before;
    bool changed = false;

    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    before = g.snapshot;
    if (memcmp(&before, next, sizeof(before)) != 0) {
        g.snapshot = *next;
        changed = true;
    }
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }

    if (changed) {
        ESP_LOGI(TAG, "state wifi=%d/%d eth=%d/%d",
                 next->wifi_link, next->wifi_ip, next->eth_link, next->eth_ip);
        notify_subscribers(next);
    }
}

static void event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    net_manager_snapshot_t next = net_manager_snapshot();

    if (base == WIFI_EVENT) {
        if (id == WIFI_EVENT_STA_CONNECTED) {
            next.wifi_link = true;
        } else if (id == WIFI_EVENT_STA_DISCONNECTED) {
            next.wifi_link = false;
            next.wifi_ip = false;
            next.wifi_ip_addr[0] = '\0';
        }
    } else if (base == IP_EVENT) {
        if (id == IP_EVENT_STA_GOT_IP) {
            ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
            next.wifi_link = true;
            next.wifi_ip = true;
            if (event) {
                snprintf(next.wifi_ip_addr, sizeof(next.wifi_ip_addr), IPSTR,
                         IP2STR(&event->ip_info.ip));
            }
        } else if (id == IP_EVENT_STA_LOST_IP) {
            next.wifi_ip = false;
            next.wifi_ip_addr[0] = '\0';
        }
#ifdef CONFIG_NETWORK_USE_ETHERNET
        else if (id == IP_EVENT_ETH_GOT_IP) {
            ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
            next.eth_link = true;
            next.eth_ip = true;
            if (event) {
                snprintf(next.eth_ip_addr, sizeof(next.eth_ip_addr), IPSTR,
                         IP2STR(&event->ip_info.ip));
            }
        }
#endif
    }
#ifdef CONFIG_NETWORK_USE_ETHERNET
    else if (base == ETH_EVENT) {
        if (id == ETHERNET_EVENT_CONNECTED) {
            next.eth_link = true;
        } else if (id == ETHERNET_EVENT_DISCONNECTED || id == ETHERNET_EVENT_STOP) {
            next.eth_link = false;
            next.eth_ip = false;
            next.eth_ip_addr[0] = '\0';
        }
    }
#endif

    update_snapshot(&next);
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        provisioning_mode_note_wifi_failure(&next);
    } else if (base == IP_EVENT &&
               (id == IP_EVENT_STA_GOT_IP
#ifdef CONFIG_NETWORK_USE_ETHERNET
                || id == IP_EVENT_ETH_GOT_IP
#endif
               )) {
        provisioning_mode_note_network_up();
    }
}

void net_manager_init(void)
{
    if (g.initialized) {
        return;
    }
    ensure_lock();
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, ESP_EVENT_ANY_ID, event_handler, NULL));
#ifdef CONFIG_NETWORK_USE_ETHERNET
    ESP_ERROR_CHECK(esp_event_handler_register(ETH_EVENT, ESP_EVENT_ANY_ID, event_handler, NULL));
#endif
    g.initialized = true;
    // Wi-Fi may already have obtained its IP while the desktop was starting.
    esp_netif_t *sta = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    esp_netif_ip_info_t ip = {0};
    if (sta && esp_netif_get_ip_info(sta, &ip) == ESP_OK && ip.ip.addr) {
        net_manager_snapshot_t next = net_manager_snapshot();
        next.wifi_link = next.wifi_ip = true;
        snprintf(next.wifi_ip_addr, sizeof(next.wifi_ip_addr), IPSTR, IP2STR(&ip.ip));
        update_snapshot(&next);
    }
}

void net_manager_subscribe(net_manager_cb_t cb, void *user)
{
    if (!cb) {
        return;
    }
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    for (size_t i = 0; i < NET_MANAGER_MAX_SUBS; ++i) {
        if (!g.subs[i].cb) {
            g.subs[i].cb = cb;
            g.subs[i].user = user;
            break;
        }
    }
    net_manager_snapshot_t snapshot = g.snapshot;
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }
    cb(&snapshot, user);
}

void net_manager_note_legacy_network(bool connected)
{
    net_manager_snapshot_t next = net_manager_snapshot();
#ifdef CONFIG_NETWORK_USE_ETHERNET
    next.eth_link = connected;
    next.eth_ip = connected;
#else
    next.wifi_link = connected;
    next.wifi_ip = connected;
#endif
    update_snapshot(&next);
}

net_manager_snapshot_t net_manager_snapshot(void)
{
    net_manager_snapshot_t snapshot = {0};
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
        snapshot = g.snapshot;
        xSemaphoreGive(g.lock);
    }
    return snapshot;
}

bool net_manager_connected(void)
{
    net_manager_snapshot_t snapshot = net_manager_snapshot();
    return snapshot.wifi_ip || snapshot.eth_ip;
}
