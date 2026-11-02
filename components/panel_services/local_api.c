#include "local_api.h"

#include "esp_app_desc.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "bsp/display.h"
#include "common.h"
#include "cjson_tiny.h"
#include "lcd_renderer.h"
#include "media_mode.h"
#include "net_manager.h"
#include "nvs_config.h"
#include "nyx_creds.h"
#include "nyx_node.h"
#include "qr_scanner.h"
#include "snapshot.h"
#include "settings.h"
#include "stream_publisher.h"
#include "network.h"
#include "panel_ota.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "mbedtls/sha256.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>

// 0 = sticky (image / text persists until explicitly cleared or replaced).
// HOME state for the AI terminal: pushed image is the durable surface.
#define LCD_DEFAULT_DURATION_MS 0u
#define LCD_IMAGE_MAX_BYTES     (64u * 1024u)
#define WIFI_TEST_TIMEOUT_MS    60000u

static const char *TAG = "haCam.api";

static const char *header_value(const nyx_proxy_request_t *req, const char *name)
{
    if (!req || !name) {
        return NULL;
    }
    for (size_t i = 0; i < req->header_count; ++i) {
        if (req->headers[i].name && strcasecmp(req->headers[i].name, name) == 0) {
            return req->headers[i].value;
        }
    }
    return NULL;
}

static uint32_t query_u32(const char *query, const char *name, uint32_t fallback)
{
    if (!query || !name) {
        return fallback;
    }
    size_t name_len = strlen(name);
    const char *p = query;
    while (*p) {
        if (strncmp(p, name, name_len) == 0 && p[name_len] == '=') {
            char *end = NULL;
            unsigned long v = strtoul(p + name_len + 1u, &end, 10);
            if (end != p + name_len + 1u && v <= UINT32_MAX) {
                return (uint32_t)v;
            }
        }
        p = strchr(p, '&');
        if (!p) {
            break;
        }
        p++;
    }
    return fallback;
}

static char *json_dup_string(cJSON *root, const char *name)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, name);
    if (!cJSON_IsString(item) || !item->valuestring) {
        return NULL;
    }
    size_t len = strlen(item->valuestring);
    char *out = malloc(len + 1u);
    if (!out) {
        return NULL;
    }
    memcpy(out, item->valuestring, len + 1u);
    return out;
}

static uint32_t json_u32(cJSON *root, const char *name, uint32_t fallback)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, name);
    if (cJSON_IsNumber(item) && item->valuedouble >= 0.0 && item->valuedouble <= UINT32_MAX) {
        return (uint32_t)item->valuedouble;
    }
    return fallback;
}

static bool json_bool(cJSON *root, const char *name, bool fallback)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, name);
    if (cJSON_IsBool(item)) {
        return cJSON_IsTrue(item);
    }
    return fallback;
}

static void json_response_len(char *resp, size_t *resp_len)
{
    if (resp_len) {
        *resp_len = strlen(resp);
    }
}

static int object_response(cJSON *root, char *resp, size_t cap, size_t *resp_len)
{
    bool ok=root && cJSON_PrintPreallocated(root,resp,cap,false);
    cJSON_Delete(root);
    if (!ok) snprintf(resp,cap,"{\"error\":\"response_buffer_or_allocation\"}");
    json_response_len(resp,resp_len);
    return ok ? 200 : 500;
}

size_t local_api_format_status_json(char *resp, size_t cap)
{
    if (!resp || cap == 0u) {
        return 0u;
    }

    const char *nyx = "down";
    if (nyx_node_state() == NYX_NODE_UP) {
        nyx = "up";
    } else if (nyx_node_state() == NYX_NODE_CONNECTING) {
        nyx = "connecting";
    }

    const esp_partition_t *running = esp_ota_get_running_partition();
    net_manager_snapshot_t net = net_manager_snapshot();
    cfg_wifi_t wifi = {0};
    (void)nvs_config_get_wifi(&wifi);
    stream_publisher_status_t stream = {0};
    stream_publisher_get_status(&stream);
    media_mode_t media_mode = media_mode_current();

    cJSON *root = cJSON_CreateObject();
    if (!root) return 0;
    cJSON_AddStringToObject(root, "fw", esp_app_get_description()->version);
    cJSON_AddStringToObject(root, "project", esp_app_get_description()->project_name);
    char app_sha[65];
    for (size_t i=0;i<32;++i) snprintf(app_sha+i*2,3,"%02x",esp_app_get_description()->app_elf_sha256[i]);
    cJSON_AddStringToObject(root, "app_sha256", app_sha);
    cJSON_AddStringToObject(root, "board", "Waveshare ESP32-P4 JD9365 / OV5647 / ES8311");
    cJSON_AddNumberToObject(root, "uptime_s", esp_timer_get_time() / 1000000ULL);
    cJSON *network = cJSON_AddObjectToObject(root, "net");
    cJSON_AddBoolToObject(network, "connected", net.wifi_ip || net.eth_ip);
    cJSON *w = cJSON_AddObjectToObject(network, "wifi");
    wifi_config_t station = {0};
    if (esp_wifi_get_config(WIFI_IF_STA, &station) == ESP_OK && !wifi.ssid_1[0])
        snprintf(wifi.ssid_1, sizeof(wifi.ssid_1), "%.*s", 32, station.sta.ssid);
    cJSON_AddStringToObject(w, "ssid", wifi.ssid_1);
    cJSON_AddStringToObject(w, "ip", net.wifi_ip_addr);
    cJSON *eth = cJSON_AddObjectToObject(network, "eth");
    cJSON_AddStringToObject(eth, "ip", net.eth_ip_addr);
    cJSON *node = cJSON_AddObjectToObject(root, "nyx");
    cJSON_AddStringToObject(node, "state", nyx);
    cJSON_AddStringToObject(node, "slug", nyx_creds_service_slug());
    cJSON_AddStringToObject(node, "last_error", nyx_node_last_error());
    cJSON *ota = cJSON_AddObjectToObject(root, "ota");
    cJSON_AddStringToObject(ota, "running", running ? running->label : "");
    const esp_partition_t *next = esp_ota_get_next_update_partition(NULL);
    cJSON_AddStringToObject(ota, "next", next ? next->label : "");
    cJSON_AddBoolToObject(ota, "busy", panel_ota_busy());
    cJSON_AddStringToObject(ota, "stage", panel_ota_stage());
    esp_err_t update_error = panel_ota_error();
    cJSON_AddStringToObject(ota, "last_error", update_error == ESP_OK ? "" :
        update_error == ESP_ERR_INVALID_CRC ? "sha256_mismatch" : esp_err_to_name(update_error));
    esp_ota_img_states_t ota_state;
    cJSON_AddStringToObject(ota, "state", running && esp_ota_get_state_partition(running, &ota_state)==ESP_OK ?
        (ota_state==ESP_OTA_IMG_VALID ? "valid" : ota_state==ESP_OTA_IMG_PENDING_VERIFY ? "pending_verify" : "other") : "initial");
    cJSON *st = cJSON_AddObjectToObject(root, "stream");
    cJSON_AddStringToObject(st, "state", stream.state);
    cJSON_AddStringToObject(st, "last_error", stream.last_error);
    cJSON *lcd = cJSON_AddObjectToObject(root, "lcd");
    const char *screen = "unknown";
    size_t depth = 0;
    (void)lcd_renderer_current_screen(&screen, &depth);
    cJSON_AddStringToObject(lcd, "mode", screen);
    cJSON_AddNumberToObject(lcd, "depth", depth);
    cJSON *mqtt = cJSON_AddObjectToObject(root, "mqtt");
    cJSON_AddStringToObject(mqtt, "state", "disabled");
    cJSON *media = cJSON_AddObjectToObject(root, "media");
    cJSON_AddStringToObject(media, "mode", media_mode_name(media_mode));
    cJSON *heap = cJSON_AddObjectToObject(root, "heap");
    cJSON_AddNumberToObject(heap, "free", esp_get_free_heap_size());
    cJSON_AddNumberToObject(heap, "spiram_free", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    cJSON_AddNumberToObject(heap, "dma_free", heap_caps_get_free_size(MALLOC_CAP_DMA));
    cJSON_AddNumberToObject(heap, "dma_largest_free", heap_caps_get_largest_free_block(MALLOC_CAP_DMA));
    cJSON_AddNumberToObject(heap, "internal_free", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    cJSON_AddNumberToObject(heap, "internal_largest_free", heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    bool ok = cJSON_PrintPreallocated(root, resp, cap, false);
    cJSON_Delete(root);
    if (!ok) snprintf(resp, cap, "{\"error\":\"status_buffer_too_small\"}");
    return strlen(resp);
}

static bool valid_config_fields(const cJSON *root)
{
    static const struct { const char *name; double min, max; } numbers[] = {
        {"fps",1,60}, {"bitrate",32000,8000000}, {"brightness",0,100},
        {"rotation",0,3}, {"port",1,65535},
    };
    for (size_t i=0;i<sizeof(numbers)/sizeof(numbers[0]);++i) {
        const cJSON *v=cJSON_GetObjectItemCaseSensitive(root,numbers[i].name);
        if (v && (!cJSON_IsNumber(v) || !isfinite(v->valuedouble) ||
            v->valuedouble<numbers[i].min || v->valuedouble>numbers[i].max ||
            v->valuedouble!=(double)(uint32_t)v->valuedouble)) return false;
    }
    static const struct { const char *name; size_t cap; } strings[] = {
        {"target_whip_url",CFG_STREAM_TARGET_MAX}, {"idle_mode",CFG_LCD_MODE_MAX},
        {"label",CFG_DEVICE_TEXT_MAX}, {"location",CFG_DEVICE_TEXT_MAX},
        {"ssid_1",CFG_WIFI_SSID_MAX}, {"ssid_2",CFG_WIFI_SSID_MAX},
        {"psw_1",CFG_WIFI_PSW_MAX}, {"psw_2",CFG_WIFI_PSW_MAX},
        {"host",CFG_MQTT_HOST_MAX}, {"user",CFG_MQTT_USER_MAX},
        {"pass",CFG_MQTT_PASS_MAX}, {"base_topic",CFG_MQTT_TOPIC_MAX},
    };
    for (size_t i=0;i<sizeof(strings)/sizeof(strings[0]);++i) {
        const cJSON *v=cJSON_GetObjectItemCaseSensitive(root,strings[i].name);
        if (v && (!cJSON_IsString(v) || strlen(v->valuestring)>=strings[i].cap)) return false;
    }
    const char *booleans[]={"enabled","auto_start","dry_run"};
    for (size_t i=0;i<sizeof(booleans)/sizeof(booleans[0]);++i) {
        const cJSON *v=cJSON_GetObjectItemCaseSensitive(root,booleans[i]);
        if (v && !cJSON_IsBool(v)) return false;
    }
    const cJSON *url=cJSON_GetObjectItemCaseSensitive(root,"target_whip_url");
    if (url && *url->valuestring && strncmp(url->valuestring,"http://",7) && strncmp(url->valuestring,"https://",8)) return false;
    return true;
}

static cJSON *parse_json_body(const nyx_proxy_request_t *req, char *resp, size_t cap,
                              size_t *resp_len, int *status)
{
    if (status) {
        *status = 400;
    }
    if (!req->body || req->body_len == 0u) {
        snprintf(resp, cap, "{\"error\":\"empty_json_body\"}");
        json_response_len(resp, resp_len);
        return NULL;
    }
    char *json = malloc(req->body_len + 1u);
    if (!json) {
        if (status) {
            *status = 500;
        }
        snprintf(resp, cap, "{\"error\":\"oom\"}");
        json_response_len(resp, resp_len);
        return NULL;
    }
    memcpy(json, req->body, req->body_len);
    json[req->body_len] = '\0';
    cJSON *root = cJSON_Parse(json);
    free(json);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        snprintf(resp, cap, "{\"error\":\"invalid_json\"}");
        json_response_len(resp, resp_len);
        return NULL;
    }
    if (req->path && !strncmp(req->path,"/config/",8) && !valid_config_fields(root)) {
        cJSON_Delete(root);
        snprintf(resp,cap,"{\"error\":\"invalid_config_field\"}");
        json_response_len(resp,resp_len);
        return NULL;
    }
    if (status) {
        *status = 200;
    }
    return root;
}

static int log_route_result(const nyx_proxy_request_t *req, int status,
                            const char *content_type, size_t resp_len)
{
    ESP_LOGI(TAG, "route method=%s path=%s body_len=%u status=%d resp_len=%u content_type=%s",
             req && req->method ? req->method : "",
             req && req->path ? req->path : "",
             (unsigned)(req ? req->body_len : 0u),
             status,
             (unsigned)resp_len,
             content_type ? content_type : "");
    return status;
}

static void json_copy_string(cJSON *root, const char *name, char *dst, size_t cap)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, name);
    if (cJSON_IsString(item) && item->valuestring && cap > 0u) {
        snprintf(dst, cap, "%s", item->valuestring);
    }
}

static bool valid_lcd_idle_mode(const char *mode)
{
    return mode &&
           (strcmp(mode, "qr") == 0 ||
            strcmp(mode, "preview") == 0 ||
            strcmp(mode, "blank") == 0);
}

static bool json_has_bool(cJSON *root, const char *name, bool *out)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, name);
    if (!cJSON_IsBool(item)) {
        return false;
    }
    if (out) {
        *out = cJSON_IsTrue(item);
    }
    return true;
}

static void delayed_reboot_task(void *arg)
{
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(500));
    esp_restart();
}

static void schedule_reboot(void)
{
    xTaskCreate(delayed_reboot_task, "api_reboot", 3072, NULL, 5, NULL);
}

static bool wait_for_wifi_result(bool expected_connected, uint32_t timeout_ms)
{
    int64_t deadline = esp_timer_get_time() + (int64_t)timeout_ms * 1000LL;
    while (esp_timer_get_time() < deadline) {
        if (network_is_connected() == expected_connected) {
            return true;
        }
        vTaskDelay(pdMS_TO_TICKS(250));
    }
    return network_is_connected() == expected_connected;
}

static bool test_wifi_config(const cfg_wifi_t *candidate, const cfg_wifi_t *previous)
{
    if (!candidate || candidate->ssid_1[0] == '\0') {
        return false;
    }
    bool was_connected = network_is_connected();
    network_connect_wifi(candidate->ssid_1, candidate->psw_1);
    bool ok = wait_for_wifi_result(true, WIFI_TEST_TIMEOUT_MS);
    if (!ok && candidate->ssid_2[0] != '\0') {
        network_connect_wifi(candidate->ssid_2, candidate->psw_2);
        ok = wait_for_wifi_result(true, WIFI_TEST_TIMEOUT_MS);
    }
    if (!ok && previous) {
        network_connect_wifi(previous->ssid_1, previous->psw_1);
        if (was_connected) {
            (void)wait_for_wifi_result(true, WIFI_TEST_TIMEOUT_MS);
        }
    }
    return ok;
}

static bool dry_run_wifi_config(const cfg_wifi_t *candidate)
{
    if (!candidate || candidate->ssid_1[0] == '\0') {
        return false;
    }

    wifi_ap_record_t ap = {0};
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK &&
        strncmp((const char *)ap.ssid, candidate->ssid_1, sizeof(ap.ssid)) == 0) {
        return true;
    }

    wifi_scan_config_t scan_cfg = {
        .ssid = (uint8_t *)candidate->ssid_1,
        .show_hidden = true,
    };
    if (esp_wifi_scan_start(&scan_cfg, true) != ESP_OK) {
        return false;
    }
    uint16_t count = 0;
    if (esp_wifi_scan_get_ap_num(&count) != ESP_OK || count == 0u) {
        return false;
    }
    return true;
}

static bool parse_hex_sha256(const char *hex, uint8_t out[32])
{
    if (!hex || strlen(hex) != 64u || !out) {
        return false;
    }
    for (size_t i = 0; i < 32u; ++i) {
        char byte_str[3] = { hex[i * 2u], hex[i * 2u + 1u], '\0' };
        char *end = NULL;
        unsigned long v = strtoul(byte_str, &end, 16);
        if (!end || *end != '\0' || v > 255u) {
            return false;
        }
        out[i] = (uint8_t)v;
    }
    return true;
}

static bool sha256_matches(const uint8_t a[32], const uint8_t b[32])
{
    uint8_t diff = 0u;
    for (size_t i = 0; i < 32u; ++i) {
        diff |= (uint8_t)(a[i] ^ b[i]);
    }
    return diff == 0u;
}

static int h_status(const nyx_proxy_request_t *req, char *resp, size_t cap,
                    char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    if (resp_len) {
        *resp_len = local_api_format_status_json(resp, cap);
    }
    return 200;
}

static int h_lcd_image(const nyx_proxy_request_t *req, char *resp, size_t cap,
                       char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    ESP_LOGI(TAG, "lcd_image: handler begin body_len=%u stack_hwm=%lu",
             (unsigned)(req ? req->body_len : 0u),
             (unsigned long)uxTaskGetStackHighWaterMark(NULL));
    snprintf(ct, ct_cap, "application/json");
    const char *content_type = header_value(req, "content-type");
    if (content_type && strstr(content_type, "image/jpeg") == NULL) {
        int n = snprintf(resp, cap, "{\"error\":\"content_type_must_be_image_jpeg\"}");
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 415;
    }
    if (!req->body || req->body_len == 0u) {
        int n = snprintf(resp, cap, "{\"error\":\"empty_jpeg_body\"}");
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 400;
    }
    // 64 KiB cap workaround for upstream ESP-Hosted SDIO bug:
    // https://github.com/espressif/esp-hosted-mcu/issues/144
    // (assert failed: sdio_rx_get_buffer sdio_drv.c:830).
    // Bigger receives stall and reset the board. HA-side resize keeps
    // us under the cap; remove this once upstream lands a fix.
    if (req->body_len > LCD_IMAGE_MAX_BYTES) {
        int n = snprintf(resp, cap,
            "{\"error\":\"payload_too_large\",\"max_bytes\":%u,"
            "\"received\":%lu,\"hint\":\"resize+recompress before POST; "
            "see HA service template in tools/ha-snippets/resize-image.yaml\"}",
            (unsigned)LCD_IMAGE_MAX_BYTES, (unsigned long)req->body_len);
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 413;
    }

    uint32_t duration_ms = query_u32(req->query, "duration_ms", LCD_DEFAULT_DURATION_MS);
    esp_err_t err = lcd_renderer_push_image(req->body, req->body_len, duration_ms);
    if (err != ESP_OK) {
        int n = snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 400;
    }
    int n = snprintf(resp, cap, "{\"ok\":true,\"duration_ms\":%lu}",
             (unsigned long)duration_ms);
    if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
    ESP_LOGI(TAG, "lcd_image: response_sent");
    return 202;
}

static int h_lcd_text(const nyx_proxy_request_t *req, char *resp, size_t cap,
                      char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    if (!req->body || req->body_len == 0u) {
        int n = snprintf(resp, cap, "{\"error\":\"empty_json_body\"}");
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 400;
    }

    char *json = malloc(req->body_len + 1u);
    if (!json) {
        int n = snprintf(resp, cap, "{\"error\":\"oom\"}");
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 500;
    }
    memcpy(json, req->body, req->body_len);
    json[req->body_len] = '\0';

    cJSON *root = cJSON_Parse(json);
    free(json);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        int n = snprintf(resp, cap, "{\"error\":\"invalid_json\"}");
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 400;
    }

    char *text = json_dup_string(root, "text");
    char *size = json_dup_string(root, "size");
    char *color = json_dup_string(root, "color");
    char *bg = json_dup_string(root, "bg");
    char *align = json_dup_string(root, "align");
    uint32_t duration_ms = json_u32(root, "duration_ms", LCD_DEFAULT_DURATION_MS);
    cJSON_Delete(root);

    if (!text) {
        free(size);
        free(color);
        free(bg);
        free(align);
        int n = snprintf(resp, cap, "{\"error\":\"missing_text\"}");
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 400;
    }

    esp_err_t err = lcd_renderer_push_text(text, size, color, bg, align, duration_ms);
    free(text);
    free(size);
    free(color);
    free(bg);
    free(align);
    if (err != ESP_OK) {
        int n = snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return 400;
    }
    int n = snprintf(resp, cap, "{\"ok\":true,\"duration_ms\":%lu}",
             (unsigned long)duration_ms);
    if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
    return 202;
}

static int h_lcd_clear(const nyx_proxy_request_t *req, char *resp, size_t cap,
                       char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    lcd_renderer_clear();
    int n = snprintf(resp, cap, "{\"ok\":true}");
    if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
    return 202;
}

static int h_get_config_stream(const nyx_proxy_request_t *req, char *resp, size_t cap,
                               char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    stream_publisher_config_t cfg = {0};
    stream_publisher_get_config(&cfg);
    cJSON *result=cJSON_CreateObject();
    cJSON_AddStringToObject(result,"target_whip_url",cfg.target_whip_url);
    cJSON_AddBoolToObject(result,"auto_start",cfg.auto_start);
    cJSON_AddNumberToObject(result,"fps",cfg.fps);
    cJSON_AddNumberToObject(result,"bitrate",cfg.bitrate);
    return object_response(result,resp,cap,resp_len);
}

static int h_post_config_stream(const nyx_proxy_request_t *req, char *resp, size_t cap,
                                char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status=0;
    cJSON *root=parse_json_body(req,resp,cap,resp_len,&status);
    if (!root) return status;

    stream_publisher_config_t cfg = {0};
    stream_publisher_get_config(&cfg);
    char *target = json_dup_string(root, "target_whip_url");
    if (target) {
        snprintf(cfg.target_whip_url, sizeof(cfg.target_whip_url), "%s", target);
    }
    cfg.auto_start = json_bool(root, "auto_start", cfg.auto_start);
    cfg.fps = (uint8_t)json_u32(root, "fps", cfg.fps ? cfg.fps : VIDEO_FPS);
    cfg.bitrate = json_u32(root, "bitrate", cfg.bitrate ? cfg.bitrate : VIDEO_BITRATE);
    free(target);
    cJSON_Delete(root);

    esp_err_t err = stream_publisher_set_config(&cfg, true);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    snprintf(resp, cap, "{\"ok\":true}");
    json_response_len(resp, resp_len);
    return 202;
}

static int h_stream_start(const nyx_proxy_request_t *req, char *resp, size_t cap,
                          char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    esp_err_t err = stream_publisher_start();
    snprintf(resp, cap, err == ESP_OK ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"%s\"}",
             esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 202 : 400;
}

static int h_stream_stop(const nyx_proxy_request_t *req, char *resp, size_t cap,
                         char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    esp_err_t err = stream_publisher_stop();
    snprintf(resp, cap, err == ESP_OK ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"%s\"}",
             esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 202 : 400;
}

static int h_stream_status(const nyx_proxy_request_t *req, char *resp, size_t cap,
                           char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    stream_publisher_status_t status = {0};
    stream_publisher_get_status(&status);
    snprintf(resp, cap,
             "{\"state\":\"%s\",\"target_whip_url\":\"%s\",\"auto_start\":%s,"
             "\"fps\":%u,\"bitrate\":%lu,\"last_error\":\"%s\"}",
             status.state,
             status.target_whip_url,
             status.auto_start ? "true" : "false",
             (unsigned)status.fps,
             (unsigned long)status.bitrate,
             status.last_error);
    json_response_len(resp, resp_len);
    return 200;
}

static int h_snapshot(const nyx_proxy_request_t *req, char *resp, size_t cap,
                      char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "image/jpeg");
    if (!media_mode_is_preview()) {
        snprintf(ct, ct_cap, "application/json");
        snprintf(resp, cap, "{\"error\":\"snapshot_unavailable_in_publish_mode\"}");
        json_response_len(resp, resp_len);
        return 503;
    }
    size_t len = 0u;
    (void)media_sys_preview_snapshot_encode(2500u, NULL);
    esp_err_t err = snapshot_copy_latest((uint8_t *)resp, cap, &len);
    if (err != ESP_OK) {
        snprintf(ct, ct_cap, "application/json");
        snprintf(resp, cap, "{\"error\":\"snapshot_unavailable\"}");
        json_response_len(resp, resp_len);
        return 404;
    }
    if (resp_len) {
        *resp_len = len;
    }
    return 200;
}

static int h_lcd_snapshot(const nyx_proxy_request_t *req, char *resp, size_t cap,
                          char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "image/jpeg");
    size_t len = 0u;
    esp_err_t err = lcd_renderer_snapshot_jpeg((uint8_t *)resp, cap, &len);
    if (err != ESP_OK) {
        snprintf(ct, ct_cap, "application/json");
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return (err == ESP_ERR_INVALID_STATE || err == ESP_ERR_TIMEOUT) ? 503 : 500;
    }
    if (resp_len) {
        *resp_len = len;
    }
    return 200;
}

static int h_lcd_screen_get(const nyx_proxy_request_t *req, char *resp, size_t cap,
                            char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    const char *target = "unknown";
    size_t depth = 0u;
    esp_err_t err = lcd_renderer_current_screen(&target, &depth);
    snprintf(resp, cap, "{\"target\":\"%s\",\"depth\":%u,\"err\":\"%s\"}",
             target ? target : "unknown", (unsigned)depth, esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return 200;
}

static int h_lcd_screen_post(const nyx_proxy_request_t *req, char *resp, size_t cap,
                             char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    const char *query = req ? req->query : NULL;
    char target[32] = {0};
    if (!query) {
        snprintf(resp, cap, "{\"error\":\"missing_target\"}");
        json_response_len(resp, resp_len);
        return 400;
    }

    const char *p = query;
    while (*p) {
        if (strncmp(p, "target=", 7) == 0) {
            const char *value = p + 7;
            size_t len = strcspn(value, "&");
            if (len == 0u || len >= sizeof(target)) {
                snprintf(resp, cap, "{\"error\":\"invalid_target\"}");
                json_response_len(resp, resp_len);
                return 400;
            }
            memcpy(target, value, len);
            target[len] = '\0';
            break;
        }
        p = strchr(p, '&');
        if (!p) {
            break;
        }
        p++;
    }
    if (target[0] == '\0') {
        snprintf(resp, cap, "{\"error\":\"missing_target\"}");
        json_response_len(resp, resp_len);
        return 400;
    }

    esp_err_t err = lcd_renderer_navigate(target);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"ok\":false,\"target\":\"%s\",\"err\":\"%s\"}",
                 target, esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return err == ESP_ERR_NOT_FOUND ? 404 : 500;
    }
    snprintf(resp, cap, "{\"ok\":true,\"target\":\"%s\"}", target);
    json_response_len(resp, resp_len);
    return 200;
}

static int h_get_config_lcd(const nyx_proxy_request_t *req, char *resp, size_t cap,
                            char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    cfg_lcd_t cfg = {0};
    esp_err_t err = nvs_config_get_lcd(&cfg);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    cJSON *result=cJSON_CreateObject();
    cJSON_AddStringToObject(result,"idle_mode",cfg.idle_mode);
    cJSON_AddNumberToObject(result,"brightness",cfg.brightness);
    cJSON_AddNumberToObject(result,"rotation",cfg.rotation);
    return object_response(result,resp,cap,resp_len);
}

static int h_post_config_lcd(const nyx_proxy_request_t *req, char *resp, size_t cap,
                             char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status = 0;
    cJSON *root = parse_json_body(req, resp, cap, resp_len, &status);
    if (!root) {
        return status;
    }
    cfg_lcd_t cfg = {0};
    nvs_config_get_lcd(&cfg);
    json_copy_string(root, "idle_mode", cfg.idle_mode, sizeof(cfg.idle_mode));
    cfg.brightness = (uint8_t)json_u32(root, "brightness", cfg.brightness);
    cfg.rotation = (uint8_t)json_u32(root, "rotation", cfg.rotation);
    cJSON_Delete(root);
    if (!valid_lcd_idle_mode(cfg.idle_mode)) {
        snprintf(resp, cap, "{\"error\":\"idle_mode_must_be_qr_preview_or_blank\"}");
        json_response_len(resp, resp_len);
        return 400;
    }
    if (cfg.brightness > 100u) cfg.brightness = 100u;
    esp_err_t err = nvs_config_set_lcd(&cfg);
    if (err == ESP_OK) {
        (void)bsp_display_brightness_set(cfg.brightness);
        lcd_renderer_reload_config();
    }
    snprintf(resp, cap, err == ESP_OK ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"%s\"}",
             esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 202 : 500;
}

static int h_get_config_device(const nyx_proxy_request_t *req, char *resp, size_t cap,
                               char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    cfg_device_t cfg = {0};
    esp_err_t err = nvs_config_get_device(&cfg);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    cJSON *result=cJSON_CreateObject();
    cJSON_AddStringToObject(result,"label",cfg.label);
    cJSON_AddStringToObject(result,"location",cfg.location);
    return object_response(result,resp,cap,resp_len);
}

static int h_post_config_device(const nyx_proxy_request_t *req, char *resp, size_t cap,
                                char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status = 0;
    cJSON *root = parse_json_body(req, resp, cap, resp_len, &status);
    if (!root) {
        return status;
    }
    cfg_device_t cfg = {0};
    nvs_config_get_device(&cfg);
    json_copy_string(root, "label", cfg.label, sizeof(cfg.label));
    json_copy_string(root, "location", cfg.location, sizeof(cfg.location));
    cJSON_Delete(root);
    esp_err_t err = nvs_config_set_device(&cfg);
    snprintf(resp, cap, err == ESP_OK ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"%s\"}",
             esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 202 : 500;
}

static int h_get_config_wifi(const nyx_proxy_request_t *req, char *resp, size_t cap,
                             char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    cfg_wifi_t cfg = {0};
    esp_err_t err = nvs_config_get_wifi(&cfg);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    cJSON *result=cJSON_CreateObject();
    cJSON_AddStringToObject(result,"ssid_1",cfg.ssid_1);
    cJSON_AddStringToObject(result,"ssid_2",cfg.ssid_2);
    return object_response(result,resp,cap,resp_len);
}

static int h_post_config_wifi(const nyx_proxy_request_t *req, char *resp, size_t cap,
                              char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status = 0;
    cJSON *root = parse_json_body(req, resp, cap, resp_len, &status);
    if (!root) {
        return status;
    }
    cfg_wifi_t previous = {0};
    cfg_wifi_t cfg = {0};
    nvs_config_get_wifi(&previous);
    cfg = previous;
    json_copy_string(root, "ssid_1", cfg.ssid_1, sizeof(cfg.ssid_1));
    json_copy_string(root, "psw_1", cfg.psw_1, sizeof(cfg.psw_1));
    json_copy_string(root, "ssid_2", cfg.ssid_2, sizeof(cfg.ssid_2));
    json_copy_string(root, "psw_2", cfg.psw_2, sizeof(cfg.psw_2));
    bool dry_run = true;
    json_has_bool(root, "dry_run", &dry_run);
    cJSON_Delete(root);

    bool ok = dry_run ? dry_run_wifi_config(&cfg) : test_wifi_config(&cfg, &previous);
    if (!ok) {
        snprintf(resp, cap, "{\"ok\":false,\"error\":\"wifi_connect_timeout\",\"ssid_tried\":\"%s\"}",
                 cfg.ssid_1);
        json_response_len(resp, resp_len);
        return 408;
    }

    if (!dry_run) {
        esp_err_t err = nvs_config_set_wifi(&cfg);
        if (err == ESP_OK) {
            err = nvs_config_set_legacy_wifi_primary(&cfg);
        }
        if (err != ESP_OK) {
            snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
            json_response_len(resp, resp_len);
            return 500;
        }
    }

    snprintf(resp, cap, "{\"ok\":true,\"dry_run\":%s,\"ssid_tried\":\"%s\"}",
             dry_run ? "true" : "false", cfg.ssid_1);
    json_response_len(resp, resp_len);
    return 202;
}

static int h_get_config_lan(const nyx_proxy_request_t *req, char *resp, size_t cap,
                            char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    cfg_lan_t cfg = {0};
    esp_err_t err = nvs_config_get_lan(&cfg);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    snprintf(resp, cap, "{\"enabled\":%s,\"port\":%u}",
             cfg.enabled ? "true" : "false", (unsigned)cfg.port);
    json_response_len(resp, resp_len);
    return 200;
}

static int h_post_config_lan(const nyx_proxy_request_t *req, char *resp, size_t cap,
                             char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status = 0;
    cJSON *root = parse_json_body(req, resp, cap, resp_len, &status);
    if (!root) {
        return status;
    }
    cfg_lan_t cfg = {0};
    nvs_config_get_lan(&cfg);
    json_has_bool(root, "enabled", &cfg.enabled);
    uint32_t port = json_u32(root, "port", cfg.port);
    cJSON_Delete(root);
    if (port == 0u || port > UINT16_MAX) {
        snprintf(resp, cap, "{\"error\":\"port_must_be_1_65535\"}");
        json_response_len(resp, resp_len);
        return 400;
    }
    cfg.port = (uint16_t)port;
    esp_err_t err = nvs_config_set_lan(&cfg);
    snprintf(resp, cap,
             err == ESP_OK
                 ? "{\"ok\":true,\"restart_required\":true}"
                 : "{\"ok\":false,\"error\":\"%s\"}",
             esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 202 : 500;
}

static int h_get_config_mqtt(const nyx_proxy_request_t *req, char *resp, size_t cap,
                             char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    cfg_mqtt_t cfg = {0};
    esp_err_t err = nvs_config_get_mqtt(&cfg);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    cJSON *result=cJSON_CreateObject();
    cJSON_AddBoolToObject(result,"enabled",cfg.enabled);
    cJSON_AddStringToObject(result,"host",cfg.host);
    cJSON_AddNumberToObject(result,"port",cfg.port);
    cJSON_AddStringToObject(result,"user",cfg.user);
    cJSON_AddStringToObject(result,"pass",cfg.pass[0] ? "<redacted>" : "");
    cJSON_AddStringToObject(result,"base_topic",cfg.base_topic);
    return object_response(result,resp,cap,resp_len);
}

static int h_post_config_mqtt(const nyx_proxy_request_t *req, char *resp, size_t cap,
                              char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status = 0;
    cJSON *root = parse_json_body(req, resp, cap, resp_len, &status);
    if (!root) {
        return status;
    }
    cfg_mqtt_t cfg = {0};
    nvs_config_get_mqtt(&cfg);
    json_has_bool(root, "enabled", &cfg.enabled);
    json_copy_string(root, "host", cfg.host, sizeof(cfg.host));
    uint32_t port = json_u32(root, "port", cfg.port);
    json_copy_string(root, "user", cfg.user, sizeof(cfg.user));
    json_copy_string(root, "pass", cfg.pass, sizeof(cfg.pass));
    json_copy_string(root, "base_topic", cfg.base_topic, sizeof(cfg.base_topic));
    cJSON_Delete(root);
    if (port == 0u || port > UINT16_MAX) {
        snprintf(resp, cap, "{\"error\":\"port_must_be_1_65535\"}");
        json_response_len(resp, resp_len);
        return 400;
    }
    if (cfg.enabled && cfg.host[0] == '\0') {
        snprintf(resp, cap, "{\"error\":\"host_required\"}");
        json_response_len(resp, resp_len);
        return 400;
    }
    if (cfg.enabled && cfg.base_topic[0] == '\0') {
        snprintf(resp, cap, "{\"error\":\"base_topic_required\"}");
        json_response_len(resp, resp_len);
        return 400;
    }
    cfg.port = (uint16_t)port;
    esp_err_t err = nvs_config_set_mqtt(&cfg);
    snprintf(resp, cap,
             err == ESP_OK
                 ? "{\"ok\":true}"
                 : "{\"ok\":false,\"error\":\"%s\"}",
             esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 202 : 500;
}

static int h_system_reboot(const nyx_proxy_request_t *req, char *resp, size_t cap,
                           char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    snprintf(resp, cap, "{\"ok\":true,\"rebooting\":true}");
    json_response_len(resp, resp_len);
    schedule_reboot();
    return 200;
}

static int h_system_factory_reset(const nyx_proxy_request_t *req, char *resp, size_t cap,
                                  char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req;
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    esp_err_t err = nvs_config_factory_reset_preserve();
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 500;
    }
    snprintf(resp, cap, "{\"ok\":true,\"factory_reset\":true,\"rebooting\":true}");
    json_response_len(resp, resp_len);
    schedule_reboot();
    return 200;
}

typedef struct {
    char *url;
    uint8_t expected[32];
    const esp_partition_t *update;
} url_ota_job_t;

static void url_ota_task(void *arg)
{
    url_ota_job_t *job = arg;
    // URLs may contain signed query credentials; log only the target partition.
    ESP_LOGI(TAG, "URL OTA downloading target=%s", job->update->label);
    panel_ota_progress("downloading", ESP_OK);
    esp_http_client_config_t http_cfg = {
        .url = job->url,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .timeout_ms = 15000,
        .keep_alive_enable = true,
        // Use the same bounded flash write path as the raw OTA endpoint.
        .buffer_size = 4096,
    };
    esp_http_client_handle_t client = esp_http_client_init(&http_cfg);
    esp_err_t err = client ? ESP_OK : ESP_ERR_NO_MEM;
    int64_t image_len = 0;
    for (unsigned redirects = 0; err == ESP_OK; ++redirects) {
        err = esp_http_client_open(client, 0);
        if (err != ESP_OK) break;
        image_len = esp_http_client_fetch_headers(client);
        int status = esp_http_client_get_status_code(client);
        if (status == 200) break;
        if (redirects >= 3 || (status != 301 && status != 302 && status != 303 &&
                               status != 307 && status != 308)) {
            err = ESP_ERR_INVALID_RESPONSE;
            break;
        }
        err = esp_http_client_set_redirection(client);
        esp_http_client_close(client);
    }
    if (err == ESP_OK && (image_len <= 0 || image_len > job->update->size)) {
        err = ESP_ERR_INVALID_SIZE;
    }
    uint8_t *buffer = heap_caps_malloc(4096, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!buffer) err = ESP_ERR_NO_MEM;
    esp_ota_handle_t handle = 0;
    if (err == ESP_OK) err = esp_ota_begin(job->update, (size_t)image_len, &handle);
    if (err == ESP_OK) ESP_LOGI(TAG, "URL OTA slot erased, expected bytes=%lld", (long long)image_len);
    mbedtls_sha256_context hash;
    mbedtls_sha256_init(&hash);
    if (mbedtls_sha256_starts(&hash, 0) != 0) err = ESP_FAIL;
    size_t received = 0;
    int64_t deadline = esp_timer_get_time() + 300000000LL;
    while (err == ESP_OK && received < (size_t)image_len) {
        size_t wanted = (size_t)image_len - received;
        if (wanted > 4096) wanted = 4096;
        int count = esp_http_client_read(client, (char *)buffer, wanted);
        if (count <= 0 || esp_timer_get_time() > deadline) {
            err = ESP_ERR_TIMEOUT;
            break;
        }
        if (mbedtls_sha256_update(&hash, buffer, count) != 0) {
            err = ESP_FAIL;
            break;
        }
        err = esp_ota_write(handle, buffer, count);
        received += count;
        if (err == ESP_OK && received == (size_t)count) ESP_LOGI(TAG, "URL OTA first write complete");
        vTaskDelay(1);
    }
    uint8_t actual[32] = {0};
    if (err == ESP_OK) {
        panel_ota_progress("verifying", ESP_OK);
        if (mbedtls_sha256_finish(&hash, actual) != 0) err = ESP_FAIL;
        else if (!sha256_matches(job->expected, actual)) err = ESP_ERR_INVALID_CRC;
    }
    mbedtls_sha256_free(&hash);
    free(buffer);
    if (client) esp_http_client_cleanup(client);
    if (err == ESP_OK) {
        err = esp_ota_end(handle);
        handle = 0;
    }
    if (err == ESP_OK) err = esp_ota_set_boot_partition(job->update);
    if (handle) esp_ota_abort(handle);
    free(job->url);
    free(job);
    if (err == ESP_OK) {
        panel_ota_progress("rebooting", ESP_OK);
        ESP_LOGI(TAG, "URL OTA verified; rebooting");
        // Keep the update lock until reboot so no request can overwrite this slot.
        schedule_reboot();
    } else {
        panel_ota_progress("failed", err);
        panel_ota_unlock();
        ESP_LOGE(TAG, "URL OTA failed error=%s", esp_err_to_name(err));
    }
    vTaskDelete(NULL);
}

static int h_system_ota(const nyx_proxy_request_t *req, char *resp, size_t cap,
                        char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    int status = 0;
    cJSON *root = parse_json_body(req, resp, cap, resp_len, &status);
    if (!root) return status;
    char *url = json_dup_string(root, "url");
    char *sha = json_dup_string(root, "sha256");
    cJSON_Delete(root);
    uint8_t expected[32] = {0};
    bool valid = url && strlen(url) <= 2048 &&
        (!strncmp(url, "https://", 8) || !strncmp(url, "http://", 7)) &&
        parse_hex_sha256(sha, expected);
    free(sha);
    if (!valid) {
        free(url);
        snprintf(resp, cap, "{\"error\":\"missing_or_invalid_url_sha256\"}");
        json_response_len(resp, resp_len);
        return 400;
    }
    const esp_partition_t *update = esp_ota_get_next_update_partition(NULL);
    if (!update || !panel_ota_try_lock()) {
        free(url);
        snprintf(resp, cap, "{\"error\":\"ota_slot_unavailable_or_busy\"}");
        json_response_len(resp, resp_len);
        return 409;
    }
    url_ota_job_t *job = calloc(1, sizeof(*job));
    if (job) {
        job->url = url;
        job->update = update;
        memcpy(job->expected, expected, sizeof(expected));
        panel_ota_progress("queued", ESP_OK);
    }
    if (!job || xTaskCreatePinnedToCore(url_ota_task, "url_ota", 16 * 1024, job, 5, NULL, 0) != pdPASS) {
        free(job);
        free(url);
        panel_ota_progress("failed", ESP_ERR_NO_MEM);
        panel_ota_unlock();
        snprintf(resp, cap, "{\"error\":\"ota_worker_unavailable\"}");
        json_response_len(resp, resp_len);
        return 503;
    }
    // Nyx proxies have a finite response deadline. Download in a separate worker;
    // 202 acknowledges acceptance, and /status reports completion or failure.
    snprintf(resp, cap, "{\"ok\":true,\"accepted\":true,\"ota\":true}");
    json_response_len(resp, resp_len);
    return 202;
}

#if HA_CAM_DEV_QR_INJECT
static int h_dev_inject_qr_jpeg(const nyx_proxy_request_t *req, char *resp, size_t cap,
                                char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "application/json");
    if (!req->body || req->body_len == 0u) {
        snprintf(resp, cap, "{\"ok\":false,\"error\":\"empty_jpeg_body\"}");
        json_response_len(resp, resp_len);
        return 400;
    }

    esp_err_t err = qr_scanner_process_jpeg(req->body, req->body_len);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return err == ESP_ERR_NOT_FOUND ? 404 : 400;
    }

    snprintf(resp, cap, "{\"ok\":true,\"rebooting\":true}");
    json_response_len(resp, resp_len);
    return 202;
}
#endif

static const nyx_route_t s_routes[] = {
    { .path = "/status", .method = "GET", .fn = h_status, .user = NULL },
    { .path = "/lcd/image", .method = "POST", .fn = h_lcd_image, .user = NULL },
    { .path = "/lcd/text", .method = "POST", .fn = h_lcd_text, .user = NULL },
    { .path = "/lcd/clear", .method = "POST", .fn = h_lcd_clear, .user = NULL },
    { .path = "/lcd/snapshot.jpg", .method = "GET", .fn = h_lcd_snapshot, .user = NULL },
    { .path = "/snapshot.jpg", .method = "GET", .fn = h_snapshot, .user = NULL },
    { .path = "/stream/start", .method = "POST", .fn = h_stream_start, .user = NULL },
    { .path = "/stream/stop", .method = "POST", .fn = h_stream_stop, .user = NULL },
    { .path = "/stream/status", .method = "GET", .fn = h_stream_status, .user = NULL },
    { .path = "/lcd/screen", .method = "GET", .fn = h_lcd_screen_get, .user = NULL },
    { .path = "/lcd/screen", .method = "POST", .fn = h_lcd_screen_post, .user = NULL },
    { .path = "/config/stream", .method = "GET", .fn = h_get_config_stream, .user = NULL },
    { .path = "/config/stream", .method = "POST", .fn = h_post_config_stream, .user = NULL },
    { .path = "/config/lcd", .method = "GET", .fn = h_get_config_lcd, .user = NULL },
    { .path = "/config/lcd", .method = "POST", .fn = h_post_config_lcd, .user = NULL },
    { .path = "/config/device", .method = "GET", .fn = h_get_config_device, .user = NULL },
    { .path = "/config/device", .method = "POST", .fn = h_post_config_device, .user = NULL },
    { .path = "/config/wifi", .method = "GET", .fn = h_get_config_wifi, .user = NULL },
    { .path = "/config/wifi", .method = "POST", .fn = h_post_config_wifi, .user = NULL },
    { .path = "/config/lan_server", .method = "GET", .fn = h_get_config_lan, .user = NULL },
    { .path = "/config/lan_server", .method = "POST", .fn = h_post_config_lan, .user = NULL },
    { .path = "/config/mqtt", .method = "GET", .fn = h_get_config_mqtt, .user = NULL },
    { .path = "/config/mqtt", .method = "POST", .fn = h_post_config_mqtt, .user = NULL },
    { .path = "/system/reboot", .method = "POST", .fn = h_system_reboot, .user = NULL },
    { .path = "/system/factory_reset", .method = "POST", .fn = h_system_factory_reset, .user = NULL },
    { .path = "/system/ota", .method = "POST", .fn = h_system_ota, .user = NULL },
#if HA_CAM_DEV_QR_INJECT
    { .path = "/_dev/qr/inject_jpeg", .method = "POST", .fn = h_dev_inject_qr_jpeg, .user = NULL },
#endif
};

static const nyx_route_t s_lan_routes[] = {
    { .path = "/lcd/image", .method = "POST", .fn = h_lcd_image, .user = NULL },
    { .path = "/lcd/clear", .method = "POST", .fn = h_lcd_clear, .user = NULL },
    { .path = "/stream/start", .method = "POST", .fn = h_stream_start, .user = NULL },
    { .path = "/stream/stop", .method = "POST", .fn = h_stream_stop, .user = NULL },
    { .path = "/stream/status", .method = "GET", .fn = h_stream_status, .user = NULL },
    { .path = "/config/stream", .method = "GET", .fn = h_get_config_stream, .user = NULL },
    { .path = "/config/stream", .method = "POST", .fn = h_post_config_stream, .user = NULL },
    { .path = "/config/mqtt", .method = "GET", .fn = h_get_config_mqtt, .user = NULL },
    { .path = "/config/mqtt", .method = "POST", .fn = h_post_config_mqtt, .user = NULL },
};

void local_api_init(void)
{
}

const nyx_route_t *local_api_routes(size_t *count)
{
    if (count) {
        *count = sizeof(s_routes) / sizeof(s_routes[0]);
    }
    return s_routes;
}

const nyx_route_t *local_api_lan_routes(size_t *count)
{
    if (count) {
        *count = sizeof(s_lan_routes) / sizeof(s_lan_routes[0]);
    }
    return s_lan_routes;
}

int local_api_route_with_logging(const nyx_route_t *route,
                                 const nyx_proxy_request_t *req,
                                 char *resp_body, size_t resp_cap,
                                 char *content_type, size_t ct_cap,
                                 size_t *resp_len)
{
    if (!route || !route->fn) {
        return log_route_result(req, 404, content_type, resp_len ? *resp_len : 0u);
    }
    int status = route->fn(req, resp_body, resp_cap, content_type, ct_cap, resp_len, route->user);
    return log_route_result(req, status, content_type, resp_len ? *resp_len : 0u);
}
