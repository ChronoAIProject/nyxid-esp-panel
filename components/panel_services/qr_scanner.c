#include "qr_scanner.h"

#include "esp_heap_caps.h"
#include "esp_jpeg_dec.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "common.h"
#include "lcd_renderer.h"
#include "nvs_config.h"
#include "nyx_creds.h"
#include "provisioning.h"
#include "quirc.h"
#include "snapshot.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "haCam.qr";

#define QR_SCAN_MAX_JPEG_BYTES (80u * 1024u)
#define QR_SCAN_MAX_RGB_BYTES  (320u * 240u * 2u)
#define QR_SCAN_INTERVAL_US    (1000000LL)
#define QR_FIELD_MAX           128u

typedef struct {
    char ssid[CFG_WIFI_SSID_MAX];
    char psw[CFG_WIFI_PSW_MAX];
    char token[QR_FIELD_MAX];
    char node[QR_FIELD_MAX];
    cfg_mqtt_t mqtt;
    bool has_wifi;
    bool has_nyx;
    bool has_mqtt;
} qr_provision_t;

static struct {
    uint8_t *jpeg;
    uint8_t *rgb;
    int64_t next_scan_us;
    bool started;
    bool logged_frame;
} g;

static bool is_uuid_shape(const char *text)
{
    static const int dash_pos[] = {8, 13, 18, 23};
    if (!text || strlen(text) != 36u) {
        return false;
    }
    for (int i = 0; i < 36; ++i) {
        bool dash = false;
        for (size_t j = 0; j < sizeof(dash_pos) / sizeof(dash_pos[0]); ++j) {
            dash = dash || (i == dash_pos[j]);
        }
        if (dash) {
            if (text[i] != '-') return false;
        } else if (!isxdigit((unsigned char)text[i])) {
            return false;
        }
    }
    return true;
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool url_decode_copy(char *dst, size_t cap, const char *src, size_t len)
{
    if (!dst || cap == 0u || !src) {
        return false;
    }
    size_t out = 0u;
    for (size_t i = 0u; i < len; ++i) {
        char c = src[i];
        if (c == '+') {
            c = ' ';
        } else if (c == '%' && i + 2u < len) {
            int hi = hex_value(src[i + 1u]);
            int lo = hex_value(src[i + 2u]);
            if (hi < 0 || lo < 0) {
                return false;
            }
            c = (char)((hi << 4) | lo);
            i += 2u;
        }
        if (out + 1u >= cap) {
            return false;
        }
        dst[out++] = c;
    }
    dst[out] = '\0';
    return true;
}

static bool query_value(const char *query, const char *key, char *dst, size_t cap)
{
    if (!query || !key || !dst || cap == 0u) {
        return false;
    }
    size_t key_len = strlen(key);
    const char *p = query;
    while (*p) {
        const char *amp = strchr(p, '&');
        size_t part_len = amp ? (size_t)(amp - p) : strlen(p);
        if (part_len > key_len && strncmp(p, key, key_len) == 0 && p[key_len] == '=') {
            return url_decode_copy(dst, cap, p + key_len + 1u, part_len - key_len - 1u);
        }
        if (!amp) {
            break;
        }
        p = amp + 1u;
    }
    return false;
}

static bool parse_payload(const char *payload, qr_provision_t *out)
{
    if (!payload || !out) {
        return false;
    }
    memset(out, 0, sizeof(*out));

    const char *prefix = "nyxprov://";
    size_t prefix_len = strlen(prefix);
    if (strncmp(payload, prefix, prefix_len) != 0) {
        return false;
    }

    const char *path = payload + prefix_len;
    const char *query = strchr(path, '?');
    if (!query) {
        return false;
    }
    size_t path_len = (size_t)(query - path);
    query++;

    if (path_len == 4u && strncmp(path, "wifi", path_len) == 0) {
        out->has_wifi = query_value(query, "ssid", out->ssid, sizeof(out->ssid)) &&
                        query_value(query, "psw", out->psw, sizeof(out->psw));
        return out->has_wifi;
    }
    if (path_len == 3u && strncmp(path, "nyx", path_len) == 0) {
        out->has_nyx = query_value(query, "token", out->token, sizeof(out->token)) &&
                       query_value(query, "node", out->node, sizeof(out->node)) &&
                       is_uuid_shape(out->node);
        return out->has_nyx;
    }
    if (path_len == 4u && strncmp(path, "mqtt", path_len) == 0) {
        char port_text[8] = "";
        nvs_config_defaults_mqtt(&out->mqtt);
        out->mqtt.enabled = true;
        out->has_mqtt = query_value(query, "host", out->mqtt.host, sizeof(out->mqtt.host)) &&
                        query_value(query, "port", port_text, sizeof(port_text)) &&
                        query_value(query, "user", out->mqtt.user, sizeof(out->mqtt.user)) &&
                        query_value(query, "pass", out->mqtt.pass, sizeof(out->mqtt.pass)) &&
                        query_value(query, "topic", out->mqtt.base_topic, sizeof(out->mqtt.base_topic));
        if (out->has_mqtt) {
            char *end = NULL;
            unsigned long port = strtoul(port_text, &end, 10);
            out->has_mqtt = end && *end == '\0' && port > 0u && port <= UINT16_MAX;
            out->mqtt.port = (uint16_t)port;
        }
        return out->has_mqtt;
    }
    if (path_len == 4u && strncmp(path, "full", path_len) == 0) {
        out->has_wifi = query_value(query, "ssid", out->ssid, sizeof(out->ssid)) &&
                        query_value(query, "psw", out->psw, sizeof(out->psw));
        out->has_nyx = query_value(query, "token", out->token, sizeof(out->token)) &&
                       query_value(query, "node", out->node, sizeof(out->node)) &&
                       is_uuid_shape(out->node);
        return out->has_wifi && out->has_nyx;
    }
    return false;
}

static esp_err_t decode_jpeg_grayscale(const uint8_t *jpeg, size_t jpeg_len,
                                       uint8_t **out_gray, int *out_w, int *out_h)
{
    if (!jpeg || jpeg_len == 0u || !out_gray || !out_w || !out_h) {
        return ESP_ERR_INVALID_ARG;
    }
    *out_gray = NULL;
    *out_w = 0;
    *out_h = 0;

    jpeg_dec_config_t cfg = DEFAULT_JPEG_DEC_CONFIG();
    cfg.output_type = JPEG_PIXEL_FORMAT_RGB565_LE;

    jpeg_dec_handle_t dec = NULL;
    if (jpeg_dec_open(&cfg, &dec) != JPEG_ERR_OK || !dec) {
        return ESP_FAIL;
    }

    jpeg_dec_io_t io = {
        .inbuf = (uint8_t *)jpeg,
        .inbuf_len = (int)jpeg_len,
    };
    jpeg_dec_header_info_t info = {0};
    jpeg_error_t jerr = jpeg_dec_parse_header(dec, &io, &info);
    if (jerr != JPEG_ERR_OK || info.width == 0 || info.height == 0) {
        jpeg_dec_close(dec);
        return ESP_ERR_INVALID_ARG;
    }

    int rgb_len = 0;
    jerr = jpeg_dec_get_outbuf_len(dec, &rgb_len);
    if (jerr != JPEG_ERR_OK || rgb_len <= 0) {
        jpeg_dec_close(dec);
        return ESP_FAIL;
    }

    uint8_t *rgb = heap_caps_aligned_alloc(16, (size_t)rgb_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb) {
        rgb = heap_caps_aligned_alloc(16, (size_t)rgb_len, MALLOC_CAP_8BIT);
    }
    if (!rgb) {
        jpeg_dec_close(dec);
        return ESP_ERR_NO_MEM;
    }

    io.outbuf = rgb;
    io.out_size = rgb_len;
    jerr = jpeg_dec_process(dec, &io);
    jpeg_dec_close(dec);
    if (jerr != JPEG_ERR_OK) {
        heap_caps_free(rgb);
        return ESP_ERR_INVALID_RESPONSE;
    }

    size_t pixels = (size_t)info.width * (size_t)info.height;
    uint8_t *gray = heap_caps_malloc(pixels, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!gray) {
        gray = heap_caps_malloc(pixels, MALLOC_CAP_8BIT);
    }
    if (!gray) {
        heap_caps_free(rgb);
        return ESP_ERR_NO_MEM;
    }

    for (size_t i = 0u; i < pixels; ++i) {
        uint16_t px = (uint16_t)rgb[i * 2u] | ((uint16_t)rgb[i * 2u + 1u] << 8);
        uint8_t r = (uint8_t)(((px >> 11) & 0x1f) << 3);
        uint8_t g8 = (uint8_t)(((px >> 5) & 0x3f) << 2);
        uint8_t b = (uint8_t)((px & 0x1f) << 3);
        gray[i] = (uint8_t)(((uint16_t)r * 77u + (uint16_t)g8 * 150u + (uint16_t)b * 29u) >> 8);
    }

    heap_caps_free(rgb);
    *out_gray = gray;
    *out_w = (int)info.width;
    *out_h = (int)info.height;
    return ESP_OK;
}

static esp_err_t decode_rgb565_grayscale(const uint8_t *rgb, uint16_t w, uint16_t h,
                                         uint8_t **out_gray, int *out_w, int *out_h)
{
    if (!rgb || w == 0 || h == 0 || !out_gray || !out_w || !out_h) {
        return ESP_ERR_INVALID_ARG;
    }
    *out_gray = NULL;
    *out_w = 0;
    *out_h = 0;

    size_t pixels = (size_t)w * (size_t)h;
    uint8_t *gray = heap_caps_malloc(pixels, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!gray) {
        gray = heap_caps_malloc(pixels, MALLOC_CAP_8BIT);
    }
    if (!gray) {
        return ESP_ERR_NO_MEM;
    }

    for (size_t i = 0u; i < pixels; ++i) {
        uint16_t px = (uint16_t)rgb[i * 2u] | ((uint16_t)rgb[i * 2u + 1u] << 8);
        uint8_t r = (uint8_t)(((px >> 11) & 0x1f) << 3);
        uint8_t g8 = (uint8_t)(((px >> 5) & 0x3f) << 2);
        uint8_t b = (uint8_t)((px & 0x1f) << 3);
        gray[i] = (uint8_t)(((uint16_t)r * 77u + (uint16_t)g8 * 150u + (uint16_t)b * 29u) >> 8);
    }

    *out_gray = gray;
    *out_w = (int)w;
    *out_h = (int)h;
    return ESP_OK;
}

static bool decode_qr_payload(const uint8_t *gray, int w, int h, char *payload, size_t cap)
{
    struct quirc *qr = quirc_new();
    if (!qr) {
        return false;
    }
    if (quirc_resize(qr, w, h) < 0) {
        quirc_destroy(qr);
        return false;
    }

    int qw = 0;
    int qh = 0;
    uint8_t *buf = quirc_begin(qr, &qw, &qh);
    if (!buf || qw != w || qh != h) {
        quirc_destroy(qr);
        return false;
    }
    memcpy(buf, gray, (size_t)w * (size_t)h);
    quirc_end(qr);

    int count = quirc_count(qr);
    for (int i = 0; i < count; ++i) {
        struct quirc_code *code = heap_caps_calloc(1, sizeof(*code), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!code) {
            code = heap_caps_calloc(1, sizeof(*code), MALLOC_CAP_8BIT);
        }
        struct quirc_data *data = heap_caps_calloc(1, sizeof(*data), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!data) {
            data = heap_caps_calloc(1, sizeof(*data), MALLOC_CAP_8BIT);
        }
        if (!code || !data) {
            heap_caps_free(code);
            heap_caps_free(data);
            break;
        }

        quirc_extract(qr, i, code);
        quirc_decode_error_t err = quirc_decode(code, data);
        if (err != QUIRC_SUCCESS) {
            quirc_flip(code);
            err = quirc_decode(code, data);
        }
        if (err == QUIRC_SUCCESS && data->payload_len > 0) {
            size_t len = (size_t)data->payload_len;
            if (len >= cap) {
                len = cap - 1u;
            }
            memcpy(payload, data->payload, len);
            payload[len] = '\0';
            heap_caps_free(code);
            heap_caps_free(data);
            quirc_destroy(qr);
            return true;
        }
        heap_caps_free(code);
        heap_caps_free(data);
    }

    quirc_destroy(qr);
    return false;
}

static void reboot_after_delay_task(void *arg)
{
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(3000));
    esp_restart();
}

static bool apply_provisioning(const qr_provision_t *prov)
{
    if (!prov) {
        return false;
    }
    if (prov->has_wifi) {
        cfg_wifi_t wifi = {0};
        nvs_config_defaults_wifi(&wifi);
        snprintf(wifi.ssid_1, sizeof(wifi.ssid_1), "%s", prov->ssid);
        snprintf(wifi.psw_1, sizeof(wifi.psw_1), "%s", prov->psw);
        wifi.ssid_2[0] = '\0';
        wifi.psw_2[0] = '\0';
        if (nvs_config_set_wifi(&wifi) != ESP_OK ||
            nvs_config_set_legacy_wifi_primary(&wifi) != ESP_OK) {
            return false;
        }
    }
    if (prov->has_nyx) {
        if (nyx_creds_save_api_key(prov->token) != 0 ||
            nyx_creds_save_node_id(prov->node) != 0) {
            return false;
        }
    }
    if (prov->has_mqtt) {
        if (nvs_config_set_mqtt(&prov->mqtt) != ESP_OK) {
            return false;
        }
    }

    const char *payload_type = (prov->has_wifi && prov->has_nyx) ? "full" :
                               (prov->has_wifi ? "wifi" :
                                (prov->has_nyx ? "nyx" : "mqtt"));
    lcd_renderer_note_qr_decoded(payload_type);
    lcd_renderer_push_text("Got config - rebooting in 3 s", "large", "#ffffff", "#14532d",
                           "center", 3000u);
    xTaskCreate(reboot_after_delay_task, "qr_reboot", 3072, NULL, 5, NULL);
    return true;
}

esp_err_t qr_scanner_process_jpeg(const uint8_t *jpeg, size_t jpeg_len)
{
    if (!jpeg || jpeg_len == 0u) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t *gray = NULL;
    int w = 0;
    int h = 0;
    esp_err_t err = decode_jpeg_grayscale(jpeg, jpeg_len, &gray, &w, &h);
    if (err != ESP_OK) {
        return err;
    }

    char payload[512] = "";
    bool decoded = decode_qr_payload(gray, w, h, payload, sizeof(payload));
    heap_caps_free(gray);
    if (!decoded) {
        ESP_LOGD(TAG, "qr decode result=not_found jpeg_len=%u image=%dx%d",
                 (unsigned)jpeg_len, w, h);
        return ESP_ERR_NOT_FOUND;
    }

    ESP_LOGI(TAG, "decoded='%s'", payload);
    qr_provision_t prov = {0};
    if (!parse_payload(payload, &prov)) {
        ESP_LOGW(TAG, "qr decode result=unsupported payload_len=%u", (unsigned)strlen(payload));
        return ESP_ERR_INVALID_ARG;
    }
    if (!apply_provisioning(&prov)) {
        ESP_LOGE(TAG, "QR provisioning apply failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "qr decode result=applied has_wifi=%d has_nyx=%d has_mqtt=%d",
             prov.has_wifi ? 1 : 0, prov.has_nyx ? 1 : 0, prov.has_mqtt ? 1 : 0);
    return ESP_OK;
}

void qr_scanner_start(void)
{
    if (g.started) {
        return;
    }
    g.jpeg = heap_caps_malloc(QR_SCAN_MAX_JPEG_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g.jpeg) {
        g.jpeg = heap_caps_malloc(QR_SCAN_MAX_JPEG_BYTES, MALLOC_CAP_8BIT);
    }
    if (!g.jpeg) {
        ESP_LOGW(TAG, "QR scanner buffer unavailable");
        return;
    }
    g.rgb = heap_caps_aligned_alloc(64, QR_SCAN_MAX_RGB_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g.rgb) {
        g.rgb = heap_caps_aligned_alloc(64, QR_SCAN_MAX_RGB_BYTES, MALLOC_CAP_8BIT);
    }
    if (!g.rgb) {
        ESP_LOGW(TAG, "QR scanner RGB buffer unavailable");
        return;
    }
    g.started = true;
    ESP_LOGI(TAG, "QR scanner ready");
}

void qr_scanner_poll(void)
{
    if (!g.started || !provisioning_mode_active()) {
        return;
    }
    int64_t now = esp_timer_get_time();
    if (now < g.next_scan_us) {
        return;
    }
    g.next_scan_us = now + QR_SCAN_INTERVAL_US;

    uint16_t w = 0;
    uint16_t h = 0;
    size_t rgb_len = 0u;
    uint32_t seq = 0u;
    if (media_sys_preview_copy_latest_rgb565(g.rgb, QR_SCAN_MAX_RGB_BYTES, &w, &h,
                                             &rgb_len, &seq) != ESP_OK) {
        size_t jpeg_len = 0u;
        (void)media_sys_snapshot_grab(0u, NULL);
        if (snapshot_copy_latest(g.jpeg, QR_SCAN_MAX_JPEG_BYTES, &jpeg_len) != ESP_OK) {
            return;
        }
        if (!g.logged_frame && jpeg_len > 0u) {
            ESP_LOGI(TAG, "qr scanner read snapshot frame len=%u", (unsigned)jpeg_len);
            g.logged_frame = true;
        }
        (void)qr_scanner_process_jpeg(g.jpeg, jpeg_len);
        return;
    }
    if (!g.logged_frame && rgb_len > 0u) {
        ESP_LOGI(TAG, "qr scanner read preview frame len=%u image=%ux%u seq=%lu",
                 (unsigned)rgb_len, (unsigned)w, (unsigned)h, (unsigned long)seq);
        g.logged_frame = true;
    }

    uint8_t *gray = NULL;
    int gray_w = 0;
    int gray_h = 0;
    if (decode_rgb565_grayscale(g.rgb, w, h, &gray, &gray_w, &gray_h) != ESP_OK) {
        return;
    }
    char payload[512] = "";
    bool decoded = decode_qr_payload(gray, gray_w, gray_h, payload, sizeof(payload));
    heap_caps_free(gray);
    if (!decoded) {
        ESP_LOGD(TAG, "qr decode result=not_found image=%dx%d", gray_w, gray_h);
        return;
    }

    ESP_LOGI(TAG, "decoded='%s'", payload);
    qr_provision_t prov = {0};
    if (!parse_payload(payload, &prov)) {
        ESP_LOGW(TAG, "qr decode result=unsupported payload_len=%u", (unsigned)strlen(payload));
        return;
    }
    if (apply_provisioning(&prov)) {
        ESP_LOGI(TAG, "qr decode result=applied has_wifi=%d has_nyx=%d has_mqtt=%d",
                 prov.has_wifi ? 1 : 0, prov.has_nyx ? 1 : 0, prov.has_mqtt ? 1 : 0);
    }
}
