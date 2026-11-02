#include "stream_publisher.h"

#include "common.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "mailbox.h"
#include "net_manager.h"
#include "nvs_config.h"
#include "stream_sm.h"

#include <stdio.h>
#include <string.h>

static const char *TAG = "haCam.stream";
static const int64_t START_TIMEOUT_US = 35000000LL;
static const int64_t RETRY_DELAY_US = 30000000LL;
static const size_t WHIP_START_MIN_DMA_LARGEST = 12288u;

typedef enum {
    STREAM_PUB_EVT_NET = 0,
    STREAM_PUB_EVT_SET_CONFIG,
    STREAM_PUB_EVT_START,
    STREAM_PUB_EVT_STOP,
    STREAM_PUB_EVT_SUSPEND,
    STREAM_PUB_EVT_RESUME,
    STREAM_PUB_EVT_WHIP_OK,
    STREAM_PUB_EVT_WHIP_ERR,
    STREAM_PUB_EVT_RETRY,
} stream_pub_evt_kind_t;

typedef struct {
    stream_pub_evt_kind_t kind;
    stream_publisher_config_t cfg;
    bool net_up;
    char error[96];
} stream_pub_evt_t;

static struct {
    mailbox_t mb;
    stream_pub_evt_t slots[16];
    stream_sm_t sm;
    stream_publisher_config_t cfg;
    char last_error[96];
    SemaphoreHandle_t lock;
    bool started;
    bool want_started;
    bool mode_suspended;
    int64_t next_retry_us;
    int64_t starting_since_us;
} g;

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (cap > 0u) {
        snprintf(dst, cap, "%s", src ? src : "");
    }
}

static const char *state_name(stream_state_t state)
{
    switch (state) {
    case STREAM_STARTING:
        return "starting";
    case STREAM_PUBLISHING:
        return "publishing";
    case STREAM_FAILED:
        return "failed";
    case STREAM_IDLE:
    default:
        return "idle";
    }
}

static const char *evt_name(stream_evt_t evt)
{
    switch (evt) {
    case STREAM_EVT_NET_UP:
        return "net_up";
    case STREAM_EVT_NET_DOWN:
        return "net_down";
    case STREAM_EVT_TARGET_SET:
        return "target_set";
    case STREAM_EVT_TARGET_CLEAR:
        return "target_clear";
    case STREAM_EVT_START:
        return "start";
    case STREAM_EVT_STOP:
        return "stop";
    case STREAM_EVT_WHIP_OK:
        return "whip_ok";
    case STREAM_EVT_WHIP_ERR:
        return "whip_err";
    case STREAM_EVT_RETRY_TIMER:
        return "retry_timer";
    default:
        return "unknown";
    }
}

static const char *pub_evt_name(stream_pub_evt_kind_t kind)
{
    switch (kind) {
    case STREAM_PUB_EVT_NET:
        return "net";
    case STREAM_PUB_EVT_SET_CONFIG:
        return "set_config";
    case STREAM_PUB_EVT_START:
        return "start";
    case STREAM_PUB_EVT_STOP:
        return "stop";
    case STREAM_PUB_EVT_SUSPEND:
        return "suspend";
    case STREAM_PUB_EVT_RESUME:
        return "resume";
    case STREAM_PUB_EVT_WHIP_OK:
        return "whip_ok";
    case STREAM_PUB_EVT_WHIP_ERR:
        return "whip_err";
    case STREAM_PUB_EVT_RETRY:
        return "retry";
    default:
        return "unknown";
    }
}

static bool wait_for_dma_largest_locked(const char *reason, size_t min_bytes, uint32_t timeout_ms)
{
    int64_t deadline = esp_timer_get_time() + ((int64_t)timeout_ms * 1000LL);
    size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_DMA);
    while (largest < min_bytes && esp_timer_get_time() < deadline) {
        vTaskDelay(pdMS_TO_TICKS(100));
        largest = heap_caps_get_largest_free_block(MALLOC_CAP_DMA);
    }
    ESP_LOGI(TAG, "dma wait reason=%s largest=%u min=%u",
             reason ? reason : "", (unsigned)largest, (unsigned)min_bytes);
    return largest >= min_bytes;
}

static void apply_stream_event(stream_evt_t evt, const char *reason)
{
    stream_state_t before = stream_sm_state(&g.sm);
    stream_sm_apply(&g.sm, evt);
    stream_state_t after = stream_sm_state(&g.sm);
    if (after == STREAM_STARTING && before != STREAM_STARTING) {
        g.starting_since_us = esp_timer_get_time();
    } else if (after != STREAM_STARTING) {
        g.starting_since_us = 0;
    }
    if (before != after) {
        ESP_LOGI(TAG, "state %s -> %s reason=%s evt=%s",
                 state_name(before), state_name(after), reason ? reason : "", evt_name(evt));
    } else {
        ESP_LOGD(TAG, "state %s unchanged reason=%s evt=%s",
                 state_name(after), reason ? reason : "", evt_name(evt));
    }
}

static void post_event(const stream_pub_evt_t *evt)
{
    if (!mailbox_post(&g.mb, evt, MAILBOX_NOWAIT)) {
        copy_str(g.last_error, sizeof(g.last_error), "stream_mailbox_full");
    }
}

static esp_err_t save_config(const stream_publisher_config_t *cfg)
{
    cfg_stream_t stored = {0};
    snprintf(stored.target_whip_url, sizeof(stored.target_whip_url), "%s", cfg->target_whip_url);
    stored.auto_start = cfg->auto_start;
    stored.fps = cfg->fps;
    stored.bitrate = cfg->bitrate;
    return nvs_config_set_stream(&stored);
}

static void load_config(void)
{
    cfg_stream_t stored = {0};
    if (nvs_config_get_stream(&stored) != ESP_OK) {
        memset(&g.cfg, 0, sizeof(g.cfg));
        g.cfg.fps = VIDEO_FPS;
        g.cfg.bitrate = VIDEO_BITRATE;
        return;
    }
    memset(&g.cfg, 0, sizeof(g.cfg));
    snprintf(g.cfg.target_whip_url, sizeof(g.cfg.target_whip_url), "%s", stored.target_whip_url);
    g.cfg.auto_start = stored.auto_start;
    g.cfg.fps = stored.fps;
    g.cfg.bitrate = stored.bitrate;
}

static bool configs_equal(const stream_publisher_config_t *a, const stream_publisher_config_t *b)
{
    if (!a || !b) {
        return false;
    }
    uint8_t a_fps = a->fps ? a->fps : VIDEO_FPS;
    uint8_t b_fps = b->fps ? b->fps : VIDEO_FPS;
    uint32_t a_bitrate = a->bitrate ? a->bitrate : VIDEO_BITRATE;
    uint32_t b_bitrate = b->bitrate ? b->bitrate : VIDEO_BITRATE;
    return a->auto_start == b->auto_start &&
           a_fps == b_fps &&
           a_bitrate == b_bitrate &&
           strcmp(a->target_whip_url, b->target_whip_url) == 0;
}

static void on_net_event(const net_manager_snapshot_t *snapshot, void *user)
{
    (void)user;
    stream_pub_evt_t evt = {
        .kind = STREAM_PUB_EVT_NET,
        .net_up = snapshot && (snapshot->wifi_ip || snapshot->eth_ip),
    };
    post_event(&evt);
}

static void maybe_start_locked(void)
{
    stream_state_t current = stream_sm_state(&g.sm);
    if (current == STREAM_STARTING || current == STREAM_PUBLISHING) {
        ESP_LOGI(TAG, "start skipped current=%s target=%s", state_name(current), g.cfg.target_whip_url);
        return;
    }
    if (g.mode_suspended) {
        ESP_LOGI(TAG, "start deferred while media mode is preview");
        return;
    }
    if (g.cfg.target_whip_url[0] == '\0') {
        apply_stream_event(STREAM_EVT_TARGET_CLEAR, "missing_target");
        copy_str(g.last_error, sizeof(g.last_error), "missing_target");
        return;
    }
    apply_stream_event(STREAM_EVT_TARGET_SET, "target_available");
    apply_stream_event(STREAM_EVT_START, "start_requested");
    if (stream_sm_state(&g.sm) != STREAM_STARTING) {
        return;
    }
    media_sys_capture_close();
    if (!wait_for_dma_largest_locked("whip_start", WHIP_START_MIN_DMA_LARGEST, 5000)) {
        copy_str(g.last_error, sizeof(g.last_error), "dma_not_ready_for_whip");
        apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
        g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
        return;
    }
    esp_err_t prep_err = media_sys_prepare_publish_capture(g.cfg.fps);
    if (prep_err != ESP_OK) {
        snprintf(g.last_error, sizeof(g.last_error), "prepare_publish:%s", esp_err_to_name(prep_err));
        apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
        g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
        return;
    }
    ESP_LOGI(TAG, "starting WHIP target=%s fps=%u bitrate=%lu",
             g.cfg.target_whip_url, (unsigned)g.cfg.fps, (unsigned long)g.cfg.bitrate);
    int ret = start_webrtc_with_video_config(g.cfg.target_whip_url, g.cfg.fps, g.cfg.bitrate);
    if (ret == 0) {
        g.last_error[0] = '\0';
    } else {
        snprintf(g.last_error, sizeof(g.last_error), "start_webrtc:%d", ret);
        apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
        g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
        media_sys_capture_close();
        vTaskDelay(pdMS_TO_TICKS(750));
    }
}

static void stop_whip_locked(const char *reason)
{
    stop_webrtc();
    apply_stream_event(STREAM_EVT_STOP, reason);
    g.next_retry_us = 0;
    media_sys_capture_close();
    vTaskDelay(pdMS_TO_TICKS(750));
}

static void fail_starting_locked(const char *reason)
{
    copy_str(g.last_error, sizeof(g.last_error), reason);
    stop_webrtc();
    media_sys_capture_close();
    apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
    g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
    vTaskDelay(pdMS_TO_TICKS(750));
}

static void handle_event(const stream_pub_evt_t *evt)
{
    ESP_LOGD(TAG, "event kind=%s net_up=%d error=%s", pub_evt_name(evt->kind),
             evt->net_up ? 1 : 0, evt->error);
    switch (evt->kind) {
    case STREAM_PUB_EVT_NET:
        apply_stream_event(evt->net_up ? STREAM_EVT_NET_UP : STREAM_EVT_NET_DOWN,
                           evt->net_up ? "network_up" : "network_down");
        if (!evt->net_up) {
            stop_webrtc();
        } else if (g.want_started || g.cfg.auto_start) {
            maybe_start_locked();
        }
        break;
    case STREAM_PUB_EVT_SET_CONFIG: {
        bool unchanged = configs_equal(&g.cfg, &evt->cfg);
        g.cfg = evt->cfg;
        if (g.cfg.fps == 0u) {
            g.cfg.fps = VIDEO_FPS;
        }
        if (g.cfg.bitrate == 0u) {
            g.cfg.bitrate = VIDEO_BITRATE;
        }
        ESP_LOGI(TAG, "config updated target=%s auto_start=%d fps=%u bitrate=%lu",
                 g.cfg.target_whip_url, g.cfg.auto_start ? 1 : 0,
                 (unsigned)g.cfg.fps, (unsigned long)g.cfg.bitrate);
        if (g.cfg.target_whip_url[0] != '\0') {
            apply_stream_event(STREAM_EVT_TARGET_SET, "config_target_set");
        } else {
            apply_stream_event(STREAM_EVT_TARGET_CLEAR, "config_target_clear");
        }
        if (unchanged) {
            ESP_LOGI(TAG, "config unchanged; stream state preserved current=%s", state_name(stream_sm_state(&g.sm)));
            break;
        }
        if (g.cfg.auto_start || g.want_started) {
            stop_webrtc();
            apply_stream_event(STREAM_EVT_STOP, "config_restart");
            g.next_retry_us = 0;
            media_sys_capture_close();
            vTaskDelay(pdMS_TO_TICKS(750));
            maybe_start_locked();
        } else {
            stop_whip_locked("config_idle");
        }
        break;
    }
    case STREAM_PUB_EVT_START:
        g.want_started = true;
        maybe_start_locked();
        break;
    case STREAM_PUB_EVT_STOP:
        g.want_started = false;
        stop_whip_locked("stop_requested");
        break;
    case STREAM_PUB_EVT_SUSPEND:
        g.mode_suspended = true;
        stop_whip_locked("mode_preview");
        copy_str(g.last_error, sizeof(g.last_error), "suspended_for_preview");
        break;
    case STREAM_PUB_EVT_RESUME:
        g.mode_suspended = false;
        if (strcmp(g.last_error, "suspended_for_preview") == 0) {
            g.last_error[0] = '\0';
        }
        if (g.want_started || g.cfg.auto_start) {
            maybe_start_locked();
        }
        break;
    case STREAM_PUB_EVT_WHIP_OK:
        apply_stream_event(STREAM_EVT_WHIP_OK, "webrtc_connected");
        g.last_error[0] = '\0';
        break;
    case STREAM_PUB_EVT_WHIP_ERR:
        if (g.mode_suspended) {
            ESP_LOGI(TAG, "WebRTC error ignored while preview-suspended reason=%s",
                     evt->error[0] ? evt->error : "webrtc_error");
            break;
        }
        copy_str(g.last_error, sizeof(g.last_error), evt->error[0] ? evt->error : "webrtc_error");
        if (stream_sm_state(&g.sm) != STREAM_STARTING &&
            stream_sm_state(&g.sm) != STREAM_PUBLISHING) {
            ESP_LOGW(TAG, "recorded WebRTC error while state=%s reason=%s",
                     state_name(stream_sm_state(&g.sm)), g.last_error);
            break;
        }
        apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
        g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
        stop_webrtc();
        media_sys_capture_close();
        vTaskDelay(pdMS_TO_TICKS(750));
        ESP_LOGI(TAG, "retry scheduled in 30000 ms reason=%s", g.last_error);
        break;
    case STREAM_PUB_EVT_RETRY:
        if (g.mode_suspended) {
            ESP_LOGI(TAG, "retry skipped while media mode is preview");
            break;
        }
        apply_stream_event(STREAM_EVT_RETRY_TIMER, "retry_timer");
        if (stream_sm_state(&g.sm) == STREAM_STARTING) {
            ESP_LOGI(TAG, "retrying WHIP target=%s", g.cfg.target_whip_url);
            media_sys_capture_close();
            if (!wait_for_dma_largest_locked("whip_retry", WHIP_START_MIN_DMA_LARGEST, 5000)) {
                copy_str(g.last_error, sizeof(g.last_error), "retry_dma_not_ready_for_whip");
                apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
                g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
                break;
            }
            esp_err_t prep_err = media_sys_prepare_publish_capture(g.cfg.fps);
            if (prep_err != ESP_OK) {
                snprintf(g.last_error, sizeof(g.last_error), "retry_prepare:%s", esp_err_to_name(prep_err));
                apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
                g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
                break;
            }
            int ret = start_webrtc_with_video_config(g.cfg.target_whip_url, g.cfg.fps, g.cfg.bitrate);
            if (ret != 0) {
                snprintf(g.last_error, sizeof(g.last_error), "retry_start:%d", ret);
                apply_stream_event(STREAM_EVT_WHIP_ERR, g.last_error);
                g.next_retry_us = esp_timer_get_time() + RETRY_DELAY_US;
                media_sys_capture_close();
                vTaskDelay(pdMS_TO_TICKS(750));
            }
        }
        break;
    }
}

void stream_publisher_init(void)
{
    if (g.started) {
        return;
    }
    memset(&g, 0, sizeof(g));
    g.lock = xSemaphoreCreateMutex();
    mailbox_init(&g.mb, g.slots, sizeof(g.slots[0]), sizeof(g.slots) / sizeof(g.slots[0]));
    stream_sm_init(&g.sm);
    load_config();
    if (g.cfg.target_whip_url[0] != '\0') {
        apply_stream_event(STREAM_EVT_TARGET_SET, "initial_target_set");
    }
    net_manager_subscribe(on_net_event, NULL);
    g.started = true;
}

esp_err_t stream_publisher_set_config(const stream_publisher_config_t *cfg, bool persist)
{
    if (!cfg) {
        return ESP_ERR_INVALID_ARG;
    }
    stream_pub_evt_t evt = {
        .kind = STREAM_PUB_EVT_SET_CONFIG,
        .cfg = *cfg,
    };
    if (persist) {
        ESP_LOGI(TAG, "persist stream config target=%s auto_start=%d fps=%u bitrate=%lu",
                 cfg->target_whip_url, cfg->auto_start ? 1 : 0,
                 (unsigned)cfg->fps, (unsigned long)cfg->bitrate);
        esp_err_t err = save_config(cfg);
        if (err != ESP_OK) {
            return err;
        }
    }
    post_event(&evt);
    return ESP_OK;
}

esp_err_t stream_publisher_get_config(stream_publisher_config_t *cfg)
{
    if (!cfg) {
        return ESP_ERR_INVALID_ARG;
    }
    *cfg = g.cfg;
    return ESP_OK;
}

esp_err_t stream_publisher_set_target(const char *target_whip_url)
{
    stream_publisher_config_t cfg = g.cfg;
    copy_str(cfg.target_whip_url, sizeof(cfg.target_whip_url), target_whip_url);
    return stream_publisher_set_config(&cfg, true);
}

esp_err_t stream_publisher_start(void)
{
    stream_pub_evt_t evt = { .kind = STREAM_PUB_EVT_START };
    post_event(&evt);
    return ESP_OK;
}

esp_err_t stream_publisher_stop(void)
{
    stream_pub_evt_t evt = { .kind = STREAM_PUB_EVT_STOP };
    post_event(&evt);
    return ESP_OK;
}

void stream_publisher_suspend_for_preview(void)
{
    stream_pub_evt_t evt = { .kind = STREAM_PUB_EVT_SUSPEND };
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    handle_event(&evt);
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }
}

void stream_publisher_resume_after_preview(void)
{
    stream_pub_evt_t evt = { .kind = STREAM_PUB_EVT_RESUME };
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    handle_event(&evt);
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }
}

void stream_publisher_poll(void)
{
    stream_pub_evt_t evt = {0};
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    while (mailbox_recv(&g.mb, &evt, MAILBOX_NOWAIT)) {
        handle_event(&evt);
    }
    if (stream_sm_should_retry(&g.sm) && g.next_retry_us > 0 && !g.mode_suspended &&
        esp_timer_get_time() >= g.next_retry_us) {
        stream_pub_evt_t retry = { .kind = STREAM_PUB_EVT_RETRY };
        handle_event(&retry);
    }
    if (stream_sm_state(&g.sm) == STREAM_STARTING && g.starting_since_us > 0 &&
        esp_timer_get_time() - g.starting_since_us >= START_TIMEOUT_US) {
        fail_starting_locked("whip_start_timeout");
    }
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }
    query_webrtc();
}

void stream_publisher_get_status(stream_publisher_status_t *status)
{
    if (!status) {
        return;
    }
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
    }
    memset(status, 0, sizeof(*status));
    copy_str(status->state, sizeof(status->state), state_name(stream_sm_state(&g.sm)));
    copy_str(status->target_whip_url, sizeof(status->target_whip_url), g.cfg.target_whip_url);
    status->auto_start = g.cfg.auto_start;
    status->fps = g.cfg.fps;
    status->bitrate = g.cfg.bitrate;
    copy_str(status->last_error, sizeof(status->last_error), g.last_error);
    if (g.lock) {
        xSemaphoreGive(g.lock);
    }
}

void stream_publisher_on_webrtc_connected(void)
{
    stream_pub_evt_t evt = { .kind = STREAM_PUB_EVT_WHIP_OK };
    post_event(&evt);
}

void stream_publisher_on_webrtc_failed(const char *reason)
{
    stream_pub_evt_t evt = { .kind = STREAM_PUB_EVT_WHIP_ERR };
    copy_str(evt.error, sizeof(evt.error), reason);
    post_event(&evt);
}
