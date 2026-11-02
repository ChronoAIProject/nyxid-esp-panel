/* WHIP signaling for VoicePresence server
 *
 * Implements esp_peer_signaling_impl_t interface.
 * Sends SDP offer via HTTP POST to /whip/offer?sessionId=xxx
 * Receives SDP answer from HTTP response body.
 */

#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_peer_signaling.h"
#include "media_lib_os.h"

#define TAG "WHIP_SIG"

typedef struct {
    esp_peer_signaling_cfg_t cfg;
    char                     target_url[256];
    char                     resource_url[256];
    char                    *response_buf;
    int                      response_len;
    int                      response_alloc;
} whip_signaling_t;

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    whip_signaling_t *sig = (whip_signaling_t *)evt->user_data;
    if (evt->event_id == HTTP_EVENT_ON_DATA && sig) {
        int new_len = sig->response_len + evt->data_len;
        if (new_len >= sig->response_alloc) {
            int new_alloc = new_len + 1024;
            char *new_buf = realloc(sig->response_buf, new_alloc);
            if (!new_buf) {
                ESP_LOGE(TAG, "Failed to realloc response buffer (%d bytes)", new_alloc);
                return ESP_FAIL;
            }
            sig->response_buf = new_buf;
            sig->response_alloc = new_alloc;
        }
        memcpy(sig->response_buf + sig->response_len, evt->data, evt->data_len);
        sig->response_len += evt->data_len;
        sig->response_buf[sig->response_len] = '\0';
    }
    return ESP_OK;
}

static void log_sdp_edges(const char *label, const uint8_t *data, int len)
{
    if (!data || len <= 0) {
        ESP_LOGI(TAG, "%s empty", label);
        return;
    }
    int first = len > 64 ? 64 : len;
    int last = len > 64 ? 64 : len;
    ESP_LOGI(TAG, "%s len=%d first='%.*s' last='%.*s'",
             label, len, first, (const char *)data,
             last, (const char *)data + len - last);
}

static int whip_signaling_start(esp_peer_signaling_cfg_t *cfg, esp_peer_signaling_handle_t *h)
{
    if (!cfg || !h) {
        ESP_LOGE(TAG, "Invalid args to whip_signaling_start");
        return ESP_PEER_ERR_INVALID_ARG;
    }

    whip_signaling_t *sig = calloc(1, sizeof(whip_signaling_t));
    if (!sig) {
        ESP_LOGE(TAG, "Failed to alloc whip_signaling_t");
        return ESP_PEER_ERR_NO_MEM;
    }
    sig->cfg = *cfg;

    if (cfg->signal_url) {
        strncpy(sig->target_url, cfg->signal_url, sizeof(sig->target_url) - 1);
    }

    ESP_LOGI(TAG, "WHIP signaling started");
    ESP_LOGI(TAG, "  target_url: %s", sig->target_url);
    ESP_LOGI(TAG, "  role: initiator (will create SDP offer)");

    *h = (esp_peer_signaling_handle_t)sig;

    esp_peer_signaling_ice_info_t ice_info = {
        .is_initiator = true,
    };
    sig->cfg.on_ice_info(&ice_info, sig->cfg.ctx);
    sig->cfg.on_connected(sig->cfg.ctx);

    return ESP_PEER_ERR_NONE;
}

static int whip_signaling_send_msg(esp_peer_signaling_handle_t h, esp_peer_signaling_msg_t *msg)
{
    whip_signaling_t *sig = (whip_signaling_t *)h;
    if (!sig || !msg) {
        return ESP_PEER_ERR_INVALID_ARG;
    }

    if (msg->type == ESP_PEER_SIGNALING_MSG_SDP) {
        if (sig->target_url[0] == '\0') {
            ESP_LOGE(TAG, "WHIP target URL is empty");
            return ESP_PEER_ERR_FAIL;
        }

        ESP_LOGI(TAG, ">>> SDP OFFER: POST %s (%d bytes)", sig->target_url, msg->size);
        log_sdp_edges("SDP offer", msg->data, msg->size);

        sig->response_len = 0;

        esp_http_client_config_t config = {
            .url = sig->target_url,
            .method = HTTP_METHOD_POST,
            .event_handler = http_event_handler,
            .user_data = sig,
            .timeout_ms = 10000,
            .buffer_size = 4096,
            .buffer_size_tx = 4096,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        if (!client) {
            ESP_LOGE(TAG, "HTTP client init FAILED");
            return ESP_PEER_ERR_FAIL;
        }

        esp_http_client_set_header(client, "Content-Type", "application/sdp");
        esp_http_client_set_post_field(client, (const char *)msg->data, msg->size);

        esp_err_t err = esp_http_client_perform(client);
        int status = esp_http_client_get_status_code(client);
        int content_len = esp_http_client_get_content_length(client);
        char *location = NULL;
        if (esp_http_client_get_header(client, "Location", &location) == ESP_OK &&
            location && location[0] != '\0') {
            strncpy(sig->resource_url, location, sizeof(sig->resource_url) - 1);
        }
        esp_http_client_cleanup(client);

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "HTTP POST FAILED: %s", esp_err_to_name(err));
            return ESP_PEER_ERR_FAIL;
        }

        ESP_LOGI(TAG, "<<< WHIP response: HTTP %d, content_length=%d, body=%d bytes",
                 status, content_len, sig->response_len);

        if (status != 201) {
            ESP_LOGE(TAG, "WHIP offer REJECTED: HTTP %d", status);
            if (sig->response_len > 0) {
                ESP_LOGE(TAG, "Response body: %.*s", sig->response_len > 200 ? 200 : sig->response_len,
                         sig->response_buf);
            }
            return ESP_PEER_ERR_FAIL;
        }

        if (sig->response_len > 0 && sig->response_buf) {
            ESP_LOGI(TAG, "SDP answer received (%d bytes), delivering to peer", sig->response_len);
            log_sdp_edges("SDP answer", (const uint8_t *)sig->response_buf, sig->response_len);

            esp_peer_signaling_msg_t answer_msg = {
                .type = ESP_PEER_SIGNALING_MSG_SDP,
                .data = (uint8_t *)sig->response_buf,
                .size = sig->response_len,
            };
            sig->cfg.on_msg(&answer_msg, sig->cfg.ctx);
        } else {
            ESP_LOGW(TAG, "WHIP 201 but empty response body");
        }

    } else if (msg->type == ESP_PEER_SIGNALING_MSG_CANDIDATE) {
        ESP_LOGD(TAG, "ICE candidate (ignored for local network): %.*s",
                 msg->size > 80 ? 80 : msg->size, (char *)msg->data);

    } else if (msg->type == ESP_PEER_SIGNALING_MSG_BYE) {
        const char *url = sig->resource_url[0] ? sig->resource_url : sig->target_url;
        if (!url || url[0] == '\0') {
            return ESP_PEER_ERR_NONE;
        }
        ESP_LOGI(TAG, ">>> BYE: DELETE %s", url);

        esp_http_client_config_t config = {
            .url = url,
            .method = HTTP_METHOD_DELETE,
            .timeout_ms = 5000,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);
        if (client) {
            esp_err_t err = esp_http_client_perform(client);
            int status = esp_http_client_get_status_code(client);
            ESP_LOGI(TAG, "<<< BYE response: HTTP %d (err=%s)", status, esp_err_to_name(err));
            esp_http_client_cleanup(client);
        }

    } else {
        ESP_LOGW(TAG, "Unknown signaling msg type: %d", msg->type);
    }

    return ESP_PEER_ERR_NONE;
}

static int whip_signaling_stop(esp_peer_signaling_handle_t h)
{
    whip_signaling_t *sig = (whip_signaling_t *)h;
    if (!sig) {
        return ESP_PEER_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "WHIP signaling stopping target=%s", sig->target_url);
    sig->cfg.on_close(sig->cfg.ctx);

    if (sig->response_buf) {
        free(sig->response_buf);
    }
    free(sig);
    ESP_LOGI(TAG, "WHIP signaling stopped");
    return ESP_PEER_ERR_NONE;
}

const esp_peer_signaling_impl_t *esp_signaling_get_whip_impl(void)
{
    static const esp_peer_signaling_impl_t impl = {
        .start = whip_signaling_start,
        .send_msg = whip_signaling_send_msg,
        .stop = whip_signaling_stop,
    };
    return &impl;
}
