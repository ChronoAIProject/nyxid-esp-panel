#include "nyx_node.h"

#include "esp_crt_bundle.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "heap_diag.h"
#include "mailbox.h"
#include "nyx_creds.h"
#include "nyx_link_sm.h"
#include "nyx_proto.h"
#include "settings.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char *TAG = "haCam.nyx";

typedef enum {
    EVT_NET_UP = 0,
    EVT_NET_DOWN,
    EVT_WSS_CONNECTED,
    EVT_WSS_MSG,
    EVT_WSS_CLOSED,
    EVT_SEND_TEXT,
    EVT_TICK,
} evt_kind_t;

typedef struct {
    evt_kind_t kind;
    char *payload;
    size_t len;
} evt_t;

typedef struct {
    char *payload;
} proxy_task_arg_t;

static struct {
    esp_websocket_client_handle_t ws;
    nyx_link_sm_t sm;
    mailbox_t mb;
    evt_t slots[16];
    const nyx_route_t *routes;
    size_t route_count;
    nyx_route_wrapper_fn route_wrapper;
    char last_error[96];
    bool started;
    int64_t last_rx_us;
    int64_t last_heartbeat_tx_us;
    uint32_t heartbeat_interval_secs;
    uint32_t idle_timeout_secs;
    char *rx_frag;
    size_t rx_frag_len;
    size_t rx_frag_cap;
    bool ws_starting;
    bool network_up;
    int64_t last_heap_log_us;
    int64_t next_ws_start_us;
} g;

#define NYX_DEFAULT_HEARTBEAT_INTERVAL_SECS 30u
#define NYX_IDLE_TIMEOUT_FLOOR_SECS 45u
#define NYX_IDLE_TIMEOUT_MULTIPLIER 3u
#define NYX_IDLE_TIMEOUT_CEILING_SECS 3600u
#define NYX_WS_SEND_TIMEOUT_MS 3000u
#define NYX_HEAP_LOG_INTERVAL_US 30000000LL

static const char *link_state_name(nyx_link_state_t state)
{
    switch (state) {
    case NYX_LINK_CONNECTING:
        return "Connecting";
    case NYX_LINK_CONNECTED:
        return "Connected";
    case NYX_LINK_RECONNECTING:
        return "Reconnecting";
    case NYX_LINK_DISCONNECTED:
    default:
        return "Disconnected";
    }
}

static const char *link_evt_name(nyx_link_evt_t evt)
{
    switch (evt) {
    case NYX_LINK_EVT_NETIF_UP:
        return "netif_up";
    case NYX_LINK_EVT_NETIF_DOWN:
        return "netif_down";
    case NYX_LINK_EVT_WSS_ACK:
        return "wss_ack";
    case NYX_LINK_EVT_WSS_ERR:
        return "wss_err";
    case NYX_LINK_EVT_IFACE_CHANGED:
        return "iface_changed";
    case NYX_LINK_EVT_RETRY_TIMER:
        return "retry_timer";
    default:
        return "unknown";
    }
}

static const char *nyx_msg_kind_name(nyx_msg_kind_t kind)
{
    switch (kind) {
    case NYX_MSG_REGISTER_OK:
        return "register_ok";
    case NYX_MSG_AUTH_OK:
        return "auth_ok";
    case NYX_MSG_AUTH_ERROR:
        return "auth_error";
    case NYX_MSG_HEARTBEAT_PING:
        return "heartbeat_ping";
    case NYX_MSG_PROXY_REQUEST:
        return "proxy_request";
    case NYX_MSG_UNKNOWN:
    default:
        return "unknown";
    }
}

static void destroy_ws(void);

static void apply_link_event(nyx_link_evt_t evt, const char *reason)
{
    nyx_link_state_t before = nyx_link_sm_state(&g.sm);
    nyx_link_sm_apply(&g.sm, evt);
    nyx_link_state_t after = nyx_link_sm_state(&g.sm);
    if (before != after) {
        ESP_LOGI(TAG, "state %s -> %s reason=%s evt=%s backoff_ms=%lu",
                 link_state_name(before), link_state_name(after),
                 reason ? reason : "", link_evt_name(evt),
                 (unsigned long)nyx_link_sm_backoff_ms(&g.sm));
    } else {
        ESP_LOGD(TAG, "state %s unchanged reason=%s evt=%s backoff_ms=%lu",
                 link_state_name(after), reason ? reason : "", link_evt_name(evt),
                 (unsigned long)nyx_link_sm_backoff_ms(&g.sm));
    }
}

static void copy_string(char *dst, size_t cap, const char *src)
{
    if (cap == 0) {
        return;
    }
    snprintf(dst, cap, "%s", src ? src : "");
}

static uint32_t compute_idle_timeout_secs(uint32_t heartbeat_secs)
{
    if (heartbeat_secs == 0u) {
        return NYX_IDLE_TIMEOUT_FLOOR_SECS;
    }
    if (heartbeat_secs > UINT32_MAX / NYX_IDLE_TIMEOUT_MULTIPLIER) {
        return NYX_IDLE_TIMEOUT_CEILING_SECS;
    }
    uint32_t timeout = heartbeat_secs * NYX_IDLE_TIMEOUT_MULTIPLIER;
    if (timeout < NYX_IDLE_TIMEOUT_FLOOR_SECS) {
        return NYX_IDLE_TIMEOUT_FLOOR_SECS;
    }
    if (timeout > NYX_IDLE_TIMEOUT_CEILING_SECS) {
        return NYX_IDLE_TIMEOUT_CEILING_SECS;
    }
    return timeout;
}

static void post_event(const evt_t *e)
{
    if (!mailbox_post(&g.mb, e, MAILBOX_NOWAIT) && e->payload) {
        free(e->payload);
    }
}

static void reset_rx_fragment(void)
{
    free(g.rx_frag);
    g.rx_frag = NULL;
    g.rx_frag_len = 0u;
    g.rx_frag_cap = 0u;
}

static bool append_rx_fragment(const esp_websocket_event_data_t *ev, evt_t *out)
{
    if (!ev || !out || !ev->data_ptr || ev->data_len <= 0 || ev->payload_len <= 0 ||
        ev->payload_offset < 0 || ev->payload_offset > ev->payload_len ||
        ev->data_len > ev->payload_len - ev->payload_offset) {
        copy_string(g.last_error, sizeof(g.last_error), "bad_ws_fragment");
        reset_rx_fragment();
        return false;
    }

    size_t total = (size_t)ev->payload_len;
    size_t offset = (size_t)ev->payload_offset;
    size_t len = (size_t)ev->data_len;
    if (offset == 0u) {
        reset_rx_fragment();
        g.rx_frag = (char *)malloc(total + 1u);
        if (!g.rx_frag) {
            copy_string(g.last_error, sizeof(g.last_error), "oom_ws_fragment");
            return false;
        }
        g.rx_frag_cap = total;
    }
    if (!g.rx_frag || g.rx_frag_cap != total || offset != g.rx_frag_len) {
        copy_string(g.last_error, sizeof(g.last_error), "ws_fragment_gap");
        reset_rx_fragment();
        return false;
    }

    memcpy(g.rx_frag + offset, ev->data_ptr, len);
    g.rx_frag_len += len;
    if (g.rx_frag_len < total) {
        return false;
    }

    g.rx_frag[total] = '\0';
    out->kind = EVT_WSS_MSG;
    out->payload = g.rx_frag;
    out->len = total;
    g.rx_frag = NULL;
    g.rx_frag_len = 0u;
    g.rx_frag_cap = 0u;
    return true;
}

static void ws_event_handler(void *handler_args, esp_event_base_t base, int32_t id, void *data)
{
    (void)handler_args;
    (void)base;
    esp_websocket_event_data_t *ev = (esp_websocket_event_data_t *)data;

    if (id == WEBSOCKET_EVENT_CONNECTED) {
        ESP_LOGI(TAG, "wss connected");
        evt_t e = { .kind = EVT_WSS_CONNECTED };
        post_event(&e);
    } else if (id == WEBSOCKET_EVENT_DATA && ev && ev->op_code == 0x01 && ev->data_len > 0) {
        g.last_rx_us = esp_timer_get_time();
        ESP_LOGD(TAG, "wss data opcode=%u chunk=%d offset=%d total=%d fin=%d",
                 (unsigned)ev->op_code, ev->data_len, ev->payload_offset, ev->payload_len,
                 ev->fin ? 1 : 0);
        evt_t e = {0};
        int total = ev->payload_len > 0 ? ev->payload_len : ev->data_len;
        if (total == ev->data_len && ev->payload_offset == 0) {
            e.kind = EVT_WSS_MSG;
            e.len = (size_t)ev->data_len;
            e.payload = (char *)malloc(e.len + 1u);
            if (!e.payload) {
                copy_string(g.last_error, sizeof(g.last_error), "oom_ws_payload");
                return;
            }
            memcpy(e.payload, ev->data_ptr, e.len);
            e.payload[e.len] = '\0';
            post_event(&e);
        } else if (append_rx_fragment(ev, &e)) {
            ESP_LOGI(TAG, "wss fragmented message reassembled len=%u", (unsigned)e.len);
            post_event(&e);
        }
    } else if (id == WEBSOCKET_EVENT_DISCONNECTED || id == WEBSOCKET_EVENT_ERROR) {
        ESP_LOGW(TAG, "wss closed event=%ld", (long)id);
        g.ws_starting = false;
        reset_rx_fragment();
        evt_t e = { .kind = EVT_WSS_CLOSED };
        post_event(&e);
    }
}

static const nyx_route_t *find_route(const char *path, const char *method)
{
    if (!path) {
        return NULL;
    }
    for (size_t i = 0; i < g.route_count; ++i) {
        const nyx_route_t *route = &g.routes[i];
        if (strcmp(route->path, path) != 0) {
            continue;
        }
        if (route->method && method && strcmp(route->method, method) != 0) {
            continue;
        }
        return route;
    }
    return NULL;
}

static void send_ws_text(char *text, size_t len)
{
    if (!text) {
        return;
    }
    evt_t e = {
        .kind = EVT_SEND_TEXT,
        .payload = text,
        .len = len,
    };
    post_event(&e);
}

static void queue_proxy_response(const char *request_id, int status, const char *content_type,
                                 const void *body, size_t body_len)
{
    size_t out_cap = body_len > 0u ? (body_len * 2u + 1024u) : 1024u;
    if (out_cap < 4096u) {
        out_cap = 4096u;
    }
    char *out = (char *)malloc(out_cap);
    if (!out) {
        copy_string(g.last_error, sizeof(g.last_error), "proxy_response_oom");
        return;
    }
    size_t n = nyx_proto_build_proxy_response(out, out_cap, request_id, status,
                                              content_type, body, body_len);
    if (n == 0u) {
        free(out);
        copy_string(g.last_error, sizeof(g.last_error), "proxy_response_too_large");
        return;
    }
    send_ws_text(out, n);
}

static void handle_proxy_request(const char *json)
{
    nyx_proxy_request_t req = {0};
    if (nyx_proto_parse_text(json, &req, NULL) != NYX_MSG_PROXY_REQUEST) {
        return;
    }

    const size_t body_cap = 96u * 1024u;
    char *body = (char *)malloc(body_cap);
    char content_type[64];
    if (!body) {
        queue_proxy_response(req.request_id ? req.request_id : "", 500,
                             "application/json", "{\"error\":\"oom\"}", strlen("{\"error\":\"oom\"}"));
        nyx_proto_free_request(&req);
        return;
    }
    body[0] = '\0';
    copy_string(content_type, sizeof(content_type), "application/json");
    size_t body_len = 0u;

    int status = 404;
    const nyx_route_t *route = find_route(req.path, req.method);
    ESP_LOGI(TAG, "proxy route method=%s path=%s body_len=%u route=%s",
             req.method ? req.method : "", req.path ? req.path : "",
             (unsigned)req.body_len, route ? "hit" : "miss");
    if (route && route->fn) {
        if (g.route_wrapper) {
            status = g.route_wrapper(route, &req, body, body_cap, content_type,
                                     sizeof(content_type), &body_len);
        } else {
            status = route->fn(&req, body, body_cap, content_type, sizeof(content_type), &body_len,
                               route->user);
        }
        if (body_len == 0u && body[0] != '\0') {
            body_len = strlen(body);
        }
    } else {
        snprintf(body, body_cap, "{\"error\":\"no route\",\"method\":\"%s\",\"path\":\"%s\"}",
                 req.method ? req.method : "", req.path ? req.path : "");
        body_len = strlen(body);
    }

    ESP_LOGI(TAG, "proxy result request_id=%s status=%d content_type=%s body_len=%u",
             req.request_id ? req.request_id : "", status, content_type, (unsigned)body_len);
    queue_proxy_response(req.request_id ? req.request_id : "", status, content_type, body, body_len);
    free(body);
    nyx_proto_free_request(&req);
}

static void proxy_task(void *arg)
{
    proxy_task_arg_t *task_arg = (proxy_task_arg_t *)arg;
    if (task_arg && task_arg->payload) {
        handle_proxy_request(task_arg->payload);
        free(task_arg->payload);
    }
    free(task_arg);
    vTaskDelete(NULL);
}

static void start_proxy_task(const char *payload)
{
    if (!payload) {
        return;
    }
    proxy_task_arg_t *arg = calloc(1, sizeof(*arg));
    if (!arg) {
        copy_string(g.last_error, sizeof(g.last_error), "proxy_task_oom");
        return;
    }
    arg->payload = strdup(payload);
    if (!arg->payload) {
        free(arg);
        copy_string(g.last_error, sizeof(g.last_error), "proxy_payload_oom");
        return;
    }
    BaseType_t ok = xTaskCreate(proxy_task, "nyx_proxy", 12 * 1024, arg, 5, NULL);
    if (ok != pdPASS) {
        free(arg->payload);
        free(arg);
        copy_string(g.last_error, sizeof(g.last_error), "proxy_task_create_failed");
    }
}

static void send_auth(void)
{
    char buf[384];
    size_t n = nyx_proto_build_auth(buf, sizeof(buf), nyx_creds_node_id(), nyx_creds_api_key());
    if (n == 0u) {
        copy_string(g.last_error, sizeof(g.last_error), "auth_frame_build_failed");
        return;
    }
    ESP_LOGI(TAG, "sending auth node=%s", nyx_creds_node_id());
    int sent = esp_websocket_client_send_text(g.ws, buf, (int)n,
                                              pdMS_TO_TICKS(NYX_WS_SEND_TIMEOUT_MS));
    if (sent < 0) {
        copy_string(g.last_error, sizeof(g.last_error), "auth_send_failed");
        apply_link_event(NYX_LINK_EVT_WSS_ERR, "auth_send_failed");
        destroy_ws();
    }
}

static void destroy_ws(void)
{
    if (!g.ws) {
        return;
    }
    esp_websocket_client_handle_t ws = g.ws;
    g.ws = NULL;
    g.ws_starting = false;
    ESP_LOGI(TAG, "destroying wss client");
    (void)esp_websocket_client_destroy(ws);
    reset_rx_fragment();
}

static void maybe_log_periodic_heap(void)
{
    int64_t now_us = esp_timer_get_time();
    if (g.last_heap_log_us == 0 ||
        now_us - g.last_heap_log_us >= NYX_HEAP_LOG_INTERVAL_US) {
        g.last_heap_log_us = now_us;
        heap_diag_log(TAG, "periodic");
    }
}

static void handle_ws_msg(const char *payload)
{
    nyx_auth_result_t auth = {0};
    nyx_msg_kind_t kind = nyx_proto_parse_text(payload, NULL, &auth);
    g.last_rx_us = esp_timer_get_time();
    if (kind == NYX_MSG_HEARTBEAT_PING) {
        ESP_LOGD(TAG, "inbound kind=%s len=%u", nyx_msg_kind_name(kind),
                 (unsigned)(payload ? strlen(payload) : 0u));
    } else {
        ESP_LOGI(TAG, "inbound kind=%s len=%u", nyx_msg_kind_name(kind),
                 (unsigned)(payload ? strlen(payload) : 0u));
    }
    switch (kind) {
    case NYX_MSG_AUTH_OK:
        apply_link_event(NYX_LINK_EVT_WSS_ACK, "auth_ok");
        g.last_error[0] = '\0';
        if (auth.heartbeat_interval_secs > 0u) {
            g.heartbeat_interval_secs = auth.heartbeat_interval_secs;
            g.idle_timeout_secs = compute_idle_timeout_secs(g.heartbeat_interval_secs);
        }
        g.last_heartbeat_tx_us = esp_timer_get_time();
        ESP_LOGI(TAG, "auth_ok heartbeat_interval=%lu idle_timeout=%lu",
                 (unsigned long)g.heartbeat_interval_secs,
                 (unsigned long)g.idle_timeout_secs);
        break;
    case NYX_MSG_REGISTER_OK:
        ESP_LOGW(TAG, "REGISTER_OK ignored - node is pre-provisioned");
        break;
    case NYX_MSG_AUTH_ERROR:
        copy_string(g.last_error, sizeof(g.last_error), auth.message ? auth.message : "auth_error");
        apply_link_event(NYX_LINK_EVT_WSS_ERR, "auth_error");
        ESP_LOGE(TAG, "auth_error: %s", g.last_error);
        destroy_ws();
        break;
    case NYX_MSG_HEARTBEAT_PING: {
        char buf[128];
        size_t n = nyx_proto_build_pong(buf, sizeof(buf), auth.timestamp);
        if (n > 0u) {
            int sent = esp_websocket_client_send_text(g.ws, buf, (int)n,
                                                      pdMS_TO_TICKS(NYX_WS_SEND_TIMEOUT_MS));
            if (sent >= 0) {
                g.last_heartbeat_tx_us = esp_timer_get_time();
                ESP_LOGD(TAG, "heartbeat_pong sent");
            } else {
                copy_string(g.last_error, sizeof(g.last_error), "heartbeat_send_failed");
                apply_link_event(NYX_LINK_EVT_WSS_ERR, "heartbeat_send_failed");
                destroy_ws();
            }
        }
        break;
    }
    case NYX_MSG_PROXY_REQUEST:
        ESP_LOGI(TAG, "proxy_request received dispatching task");
        start_proxy_task(payload);
        break;
    default:
        ESP_LOGW(TAG, "unknown nyx message: %.80s", payload ? payload : "");
        break;
    }
    nyx_proto_free_auth_result(&auth);
}

static void start_ws_if_needed(void)
{
    if (!g.network_up) {
        ESP_LOGD(TAG, "wss start skipped; network is not up");
        return;
    }
    if (g.ws && (g.ws_starting || esp_websocket_client_is_connected(g.ws))) {
        ESP_LOGD(TAG, "wss start skipped starting=%d connected=%d",
                 g.ws_starting ? 1 : 0,
                 esp_websocket_client_is_connected(g.ws) ? 1 : 0);
        return;
    }
    int64_t now_us = esp_timer_get_time();
    if (nyx_link_sm_state(&g.sm) == NYX_LINK_RECONNECTING) {
        if (g.next_ws_start_us == 0) {
            uint32_t backoff_ms = nyx_link_sm_backoff_ms(&g.sm);
            g.next_ws_start_us = now_us + (int64_t)backoff_ms * 1000LL;
        }
        if (now_us < g.next_ws_start_us) {
            return;
        }
        apply_link_event(NYX_LINK_EVT_RETRY_TIMER, "retry_timer");
        g.next_ws_start_us = 0;
    }
    if (!g.ws) {
        esp_websocket_client_config_t cfg = {
            .uri = HA_CAM_NYX_WSS_URL,
            .disable_auto_reconnect = true,
            .reconnect_timeout_ms = 10000,
            .network_timeout_ms = 5000,
            .buffer_size = 20 * 1024,
            .keep_alive_enable = true,
            .keep_alive_idle = 30,
            .keep_alive_interval = 10,
            .keep_alive_count = 3,
            .pingpong_timeout_sec = 45,
            .crt_bundle_attach = esp_crt_bundle_attach,
        };
        g.ws = esp_websocket_client_init(&cfg);
        if (!g.ws) {
            copy_string(g.last_error, sizeof(g.last_error), "ws_init_failed");
            apply_link_event(NYX_LINK_EVT_WSS_ERR, "ws_init_failed");
            return;
        }
        esp_websocket_register_events(g.ws, WEBSOCKET_EVENT_ANY, ws_event_handler, NULL);
    }
    g.last_rx_us = esp_timer_get_time();
    ESP_LOGI(TAG, "starting wss url=%s", HA_CAM_NYX_WSS_URL);
    esp_err_t err = esp_websocket_client_start(g.ws);
    if (err == ESP_OK) {
        g.ws_starting = true;
    } else {
        ESP_LOGW(TAG, "wss start failed: %s", esp_err_to_name(err));
        copy_string(g.last_error, sizeof(g.last_error), "ws_start_failed");
        apply_link_event(NYX_LINK_EVT_WSS_ERR, "ws_start_failed");
        uint32_t backoff_ms = nyx_link_sm_backoff_ms(&g.sm);
        g.next_ws_start_us = esp_timer_get_time() + (int64_t)backoff_ms * 1000LL;
    }
    heap_diag_log(TAG, err == ESP_OK ? "nyx_wss_start" : "nyx_wss_start_failed");
}

static void check_idle_timeout(void)
{
    if (nyx_link_sm_state(&g.sm) != NYX_LINK_CONNECTED || !g.ws) {
        return;
    }
    uint32_t heartbeat_secs = g.heartbeat_interval_secs ?
                              g.heartbeat_interval_secs :
                              NYX_DEFAULT_HEARTBEAT_INTERVAL_SECS;
    int64_t now_us = esp_timer_get_time();
    (void)heartbeat_secs;
    uint32_t timeout_secs = g.idle_timeout_secs ? g.idle_timeout_secs : NYX_IDLE_TIMEOUT_FLOOR_SECS;
    int64_t idle_us = now_us - g.last_rx_us;
    if (idle_us > (int64_t)timeout_secs * 1000000LL) {
        ESP_LOGI(TAG, "wss idle timeout after %lld us; reconnecting", (long long)idle_us);
        copy_string(g.last_error, sizeof(g.last_error), "ws_idle_timeout");
        apply_link_event(NYX_LINK_EVT_WSS_ERR, "ws_idle_timeout");
        destroy_ws();
        start_ws_if_needed();
    }
}

static void consumer_loop(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "nyx_loop starting");

    while (true) {
        evt_t e = {0};
        if (!mailbox_recv(&g.mb, &e, 1000)) {
            check_idle_timeout();
            start_ws_if_needed();
            maybe_log_periodic_heap();
            continue;
        }

        switch (e.kind) {
        case EVT_NET_UP:
            g.network_up = true;
            apply_link_event(NYX_LINK_EVT_NETIF_UP, "network_up");
            start_ws_if_needed();
            break;
        case EVT_NET_DOWN:
            g.network_up = false;
            apply_link_event(NYX_LINK_EVT_NETIF_DOWN, "network_down");
            destroy_ws();
            break;
        case EVT_WSS_CONNECTED:
            g.ws_starting = false;
            apply_link_event(NYX_LINK_EVT_NETIF_UP, "wss_connected");
            send_auth();
            heap_diag_log(TAG, "nyx_wss_connected");
            break;
        case EVT_WSS_MSG:
            handle_ws_msg(e.payload);
            break;
        case EVT_SEND_TEXT:
            if (g.ws && e.payload && e.len > 0u) {
                int sent = esp_websocket_client_send_text(g.ws, e.payload, (int)e.len,
                                                          pdMS_TO_TICKS(NYX_WS_SEND_TIMEOUT_MS));
                if (sent < 0) {
                    copy_string(g.last_error, sizeof(g.last_error), "ws_send_failed");
                    apply_link_event(NYX_LINK_EVT_WSS_ERR, "ws_send_failed");
                    destroy_ws();
                    start_ws_if_needed();
                }
            }
            break;
        case EVT_WSS_CLOSED:
            int64_t now_us = esp_timer_get_time();
            if (!g.ws) {
                ESP_LOGW(TAG, "ignoring duplicate wss close event");
                break;
            }
            apply_link_event(NYX_LINK_EVT_WSS_ERR, "wss_closed");
            uint32_t backoff_ms = nyx_link_sm_backoff_ms(&g.sm);
            g.next_ws_start_us = now_us + (int64_t)backoff_ms * 1000LL;
            copy_string(g.last_error, sizeof(g.last_error), "ws_closed");
            destroy_ws();
            heap_diag_log(TAG, "nyx_wss_disconnected");
            start_ws_if_needed();
            break;
        case EVT_TICK:
            break;
        }
        free(e.payload);
    }
}

void nyx_node_init(void)
{
    memset(&g, 0, sizeof(g));
    mailbox_init(&g.mb, g.slots, sizeof(g.slots[0]), sizeof(g.slots) / sizeof(g.slots[0]));
    nyx_link_sm_init(&g.sm);
}

void nyx_node_set_routes(const nyx_route_t *routes, size_t count)
{
    g.routes = routes;
    g.route_count = count;
}

void nyx_node_set_route_wrapper(nyx_route_wrapper_fn wrapper)
{
    g.route_wrapper = wrapper;
}

void nyx_node_start(void)
{
    if (g.started) {
        return;
    }
    g.started = true;
    xTaskCreate(consumer_loop, "nyx_loop", 8192, NULL, 6, NULL);
}

void nyx_node_on_network(bool connected)
{
    evt_t e = { .kind = connected ? EVT_NET_UP : EVT_NET_DOWN };
    post_event(&e);
}

nyx_node_state_t nyx_node_state(void)
{
    switch (nyx_link_sm_state(&g.sm)) {
    case NYX_LINK_CONNECTED:
        return NYX_NODE_UP;
    case NYX_LINK_CONNECTING:
    case NYX_LINK_RECONNECTING:
        return NYX_NODE_CONNECTING;
    default:
        return NYX_NODE_DOWN;
    }
}

const char *nyx_node_last_error(void)
{
    return g.last_error;
}
