/* HA camera WHIP publisher
 *
 * Publishes H264 video from ESP32-P4 camera via WHIP signaling.
 */

#include <string.h>
#include "esp_webrtc.h"
#include "media_lib_os.h"
#include "common.h"
#include "esp_log.h"
#include "esp_webrtc_defaults.h"
#include "esp_peer_default.h"
#include "esp_peer.h"
#include "esp_capture.h"
#include "esp_capture_sink.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "stream_publisher.h"

#define TAG "haCamWebRTC"

static esp_webrtc_handle_t webrtc = NULL;

static const char *peer_state_str(int type)
{
    switch (type) {
        case ESP_WEBRTC_EVENT_CONNECTED:      return "CONNECTED";
        case ESP_WEBRTC_EVENT_DISCONNECTED:   return "DISCONNECTED";
        case ESP_WEBRTC_EVENT_CONNECT_FAILED: return "CONNECT_FAILED";
        case ESP_WEBRTC_EVENT_PAIRED:         return "PAIRED";
        default:                              return "UNKNOWN";
    }
}

static int webrtc_event_handler(esp_webrtc_event_t *event, void *ctx)
{
    if (!event) return 0;

    ESP_LOGI(TAG, "WebRTC event: %s (%d) decision=%s",
             peer_state_str(event->type), event->type,
             event->type == ESP_WEBRTC_EVENT_CONNECTED ? "mark_publishing" :
             event->type == ESP_WEBRTC_EVENT_DISCONNECTED ? "mark_failed_disconnected" :
             event->type == ESP_WEBRTC_EVENT_CONNECT_FAILED ? "mark_failed_connect" :
             event->type == ESP_WEBRTC_EVENT_PAIRED ? "log_pair_addr" : "ignore");

    switch (event->type) {
        case ESP_WEBRTC_EVENT_CONNECTED:
            stream_publisher_on_webrtc_connected();
            ESP_LOGI(TAG, "=== WebRTC CONNECTED ===");
            ESP_LOGI(TAG, "  Audio: disabled");
            ESP_LOGI(TAG, "  Video: H264 %dx%d@%dfps SEND_ONLY", VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS);
            ESP_LOGI(TAG, "  Free heap: %lu, PSRAM: %lu",
                     (unsigned long)esp_get_free_heap_size(),
                     (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
            break;
        case ESP_WEBRTC_EVENT_DISCONNECTED:
            media_sys_webrtc_capture_stop();
            stream_publisher_on_webrtc_failed("disconnected");
            ESP_LOGW(TAG, "=== WebRTC DISCONNECTED ===");
            break;
        case ESP_WEBRTC_EVENT_CONNECT_FAILED:
            media_sys_webrtc_capture_stop();
            stream_publisher_on_webrtc_failed("connect_failed");
            ESP_LOGE(TAG, "=== WebRTC connection FAILED ===");
            break;
        case ESP_WEBRTC_EVENT_PAIRED:
            {
                esp_peer_handle_t peer_handle = NULL;
                esp_webrtc_get_peer_connection(webrtc, &peer_handle);
                esp_peer_addr_t addr = {};
                esp_peer_get_paired_addr(peer_handle, &addr);
                ESP_LOGI(TAG, "ICE paired with server: %d.%d.%d.%d:%d",
                         addr.ipv4[0], addr.ipv4[1], addr.ipv4[2], addr.ipv4[3], addr.port);
            }
            break;
        default:
            ESP_LOGD(TAG, "Unhandled WebRTC event: %d", event->type);
            break;
    }
    return 0;
}

int start_webrtc_with_video_config(const char *session_id, uint8_t fps, uint32_t bitrate)
{
    if (fps == 0u) {
        fps = VIDEO_FPS;
    }
    if (bitrate == 0u) {
        bitrate = VIDEO_BITRATE;
    }
    if (!network_is_connected()) {
        ESP_LOGE(TAG, "Cannot start WebRTC: WiFi not connected");
        return -1;
    }
    if (webrtc) {
        ESP_LOGW(TAG, "WebRTC already running, stopping first...");
        stop_webrtc();
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Starting WebRTC");
    ESP_LOGI(TAG, "  session: %s", session_id ? session_id : "(none)");
    ESP_LOGI(TAG, "  audio:   disabled");
    ESP_LOGI(TAG, "  video:   H264 %dx%d@%ufps SEND_ONLY bitrate=%lu",
             VIDEO_WIDTH, VIDEO_HEIGHT, (unsigned)fps, (unsigned long)bitrate);

    esp_peer_default_cfg_t peer_cfg = {
        .agent_recv_timeout = 500,
        .rtp_cfg = {
            // Retain a full IDR burst while the C6/Wi-Fi transport drains it.
            .send_pool_size = 128 * 1024,
            .send_queue_num = 128,
        },
    };

    esp_webrtc_cfg_t cfg = {
        .peer_cfg = {
            .video_info = {
                .codec = ESP_PEER_VIDEO_CODEC_H264,
                .width = VIDEO_WIDTH,
                .height = VIDEO_HEIGHT,
                .fps = fps,
            },
            .audio_info = {
                .codec = ESP_PEER_AUDIO_CODEC_NONE,
            },
            .audio_dir = ESP_PEER_MEDIA_DIR_NONE,
            .video_dir = ESP_PEER_MEDIA_DIR_SEND_ONLY,
            .enable_data_channel = DATA_CHANNEL_ENABLED,
            .no_auto_reconnect = true,
            .extra_cfg = &peer_cfg,
            .extra_size = sizeof(peer_cfg),
        },
        .signaling_cfg = {
            .signal_url = (char *)session_id,
        },
        .peer_impl = esp_peer_get_default_impl(),
        .signaling_impl = esp_signaling_get_whip_impl(),
    };

    int ret = esp_webrtc_open(&cfg, &webrtc);
    if (ret != 0) {
        ESP_LOGE(TAG, "esp_webrtc_open FAILED: %d", ret);
        return ret;
    }
    ESP_LOGI(TAG, "esp_webrtc_open OK: handle=%p", webrtc);

    // Set media provider
    esp_webrtc_media_provider_t media_provider = {};
    media_sys_get_provider(&media_provider);
    esp_webrtc_set_media_provider(webrtc, &media_provider);
    esp_webrtc_set_no_auto_capture(webrtc, true);
    esp_webrtc_set_video_bitrate(webrtc, bitrate);
    ESP_LOGI(TAG, "Media provider set: capture=%p player=%p", media_provider.capture, media_provider.player);

    // Set event handler
    esp_webrtc_set_event_handler(webrtc, webrtc_event_handler, NULL);

    ESP_LOGI(TAG, "Starting signaling; peer connection will open after ICE info arrives");

    ret = media_sys_webrtc_capture_start();
    if (ret != 0) {
        ESP_LOGE(TAG, "webrtc capture start FAILED: %d", ret);
        esp_webrtc_close(webrtc);
        webrtc = NULL;
        return ret;
    }

    // Start WebRTC
    ret = esp_webrtc_start(webrtc);
    if (ret != 0) {
        ESP_LOGE(TAG, "esp_webrtc_start FAILED: %d", ret);
        media_sys_webrtc_capture_stop();
        esp_webrtc_close(webrtc);
        webrtc = NULL;
    } else {
        ESP_LOGI(TAG, "esp_webrtc_start OK");
    }
    ESP_LOGI(TAG, "========================================");
    return ret;
}

int start_webrtc(const char *session_id)
{
    return start_webrtc_with_video_config(session_id, VIDEO_FPS, VIDEO_BITRATE);
}

void query_webrtc(void)
{
    static int64_t last_query;
    int64_t now=esp_timer_get_time();
    if (webrtc && now-last_query>=10000000LL) {
        last_query=now;
        esp_webrtc_query(webrtc);
    }
}

int stop_webrtc(void)
{
    if (webrtc) {
        esp_webrtc_handle_t handle = webrtc;
        webrtc = NULL;
        ESP_LOGI(TAG, "Stopping WebRTC %p...", handle);
        media_sys_webrtc_capture_stop();
        esp_webrtc_close(handle);
        ESP_LOGI(TAG, "WebRTC stopped");
    }
    return 0;
}
