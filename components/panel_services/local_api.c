#include "local_api.h"

#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "panel_voice.h"
#include "panel_access.h"
#include "product_support.h"
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
#include "camera_profile.h"
#include "net_manager.h"
#include "nvs_config.h"
#include "nyx_creds.h"
#include "nyx_node.h"
#include "qr_scanner.h"
#include "settings.h"
#include "network.h"
#include "panel_ota.h"
#include "panel_update.h"
#include "panel_flash.h"
#include "device_code.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>

// 0 = sticky (image / text persists until explicitly cleared or replaced).
// Temporary pushes stay in RAM; the separate default-image API persists fallback art.
#define LCD_DEFAULT_DURATION_MS 0u
#define LCD_IMAGE_MAX_BYTES     (64u * 1024u)
#define WIFI_TEST_TIMEOUT_MS    60000u

static const char *TAG = "haCam.api";
static cJSON *update_json(void);

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

static cJSON *voice_json(void)
{
    panel_voice_snapshot_t v;panel_voice_snapshot(&v);
    cJSON *j=cJSON_CreateObject();
    cJSON_AddBoolToObject(j,"configured",v.configured);cJSON_AddBoolToObject(j,"busy",v.busy);
    cJSON_AddBoolToObject(j,"active",v.active);cJSON_AddBoolToObject(j,"holding",v.holding);
    cJSON_AddStringToObject(j,"input_mode",panel_voice_input_mode_name(v.input_mode));
    cJSON_AddStringToObject(j,"preferred_input_mode",panel_voice_input_mode_name(v.preferred_input_mode));
    cJSON_AddBoolToObject(j,"microphone_muted",v.microphone_muted);
    cJSON_AddNumberToObject(j,"playback_gain",v.playback_gain);
    cJSON_AddNumberToObject(j,"volume",product_codec_volume_get());
    cJSON_AddStringToObject(j,"state",v.state);cJSON_AddStringToObject(j,"message",v.message);
    cJSON_AddStringToObject(j,"service_id",v.service_id);
    cJSON_AddStringToObject(j,"agent_id",v.agent_id);cJSON_AddStringToObject(j,"agent_name",v.agent_name);
    cJSON_AddStringToObject(j,"agent_description",v.agent_description);
    cJSON_AddStringToObject(j,"session_id",v.session_id);
    cJSON_AddBoolToObject(j,"remote_initiated",v.remote_initiated);
    cJSON_AddStringToObject(j,"remote_request_id",v.remote_request_id);cJSON_AddStringToObject(j,"remote_state",v.remote_state);
    cJSON_AddNumberToObject(j,"sent_packets",v.sent_packets);cJSON_AddNumberToObject(j,"received_packets",v.received_packets);
    cJSON_AddNumberToObject(j,"playback_ms",v.playback_ms);
    cJSON_AddNumberToObject(j,"call_seconds",v.call_seconds);
    cJSON_AddBoolToObject(j,"dialing_tone",v.dialing_tone);
    cJSON_AddBoolToObject(j,"greeting_enabled",v.greeting_enabled);
    cJSON_AddStringToObject(j,"greeting_state",v.greeting_state);
    cJSON_AddNumberToObject(j,"source_peak",v.source_peak);cJSON_AddNumberToObject(j,"output_peak",v.output_peak);
    return j;
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
    cJSON_AddBoolToObject(root, "time_valid", time(NULL) >= 1700000000);
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
    char management[80];
    if(panel_management_slug(nyx_creds_service_slug(),management,sizeof(management)))
        cJSON_AddStringToObject(node,"management_slug",management);
    cJSON_AddStringToObject(node, "last_error", nyx_node_last_error());
    cJSON_AddItemToObject(root,"update",update_json());
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
    // IDF reads otadata through an MMU mapping here, bypassing esp_flash_read.
    panel_flash_lock();
    esp_err_t state_error = running ? esp_ota_get_state_partition(running, &ota_state) : ESP_ERR_NOT_FOUND;
    panel_flash_unlock();
    cJSON_AddStringToObject(ota, "state", state_error==ESP_OK ?
        (ota_state==ESP_OTA_IMG_VALID ? "valid" : ota_state==ESP_OTA_IMG_PENDING_VERIFY ? "pending_verify" : "other") : "initial");
    cJSON *st = cJSON_AddObjectToObject(root, "stream");
    cJSON_AddStringToObject(st, "state", "disabled");
    cJSON_AddStringToObject(st, "last_error", "");
    cJSON *lcd = cJSON_AddObjectToObject(root, "lcd");
    const char *screen = "unknown";
    size_t depth = 0;
    (void)lcd_renderer_current_screen(&screen, &depth);
    cJSON_AddStringToObject(lcd, "mode", screen);
    cJSON_AddNumberToObject(lcd, "depth", depth);
    lcd_default_image_info_t art = {0};
    if (lcd_renderer_default_image_info(&art) == ESP_OK) {
        cJSON_AddStringToObject(lcd, "default_image", art.stored ? "stored" : "builtin");
        cJSON_AddNumberToObject(lcd, "default_image_bytes", art.jpeg_bytes);
    }
    lcd_temporary_image_info_t temporary = {0};
    if (lcd_renderer_temporary_image_info(&temporary) == ESP_OK) {
        cJSON *visible = cJSON_AddObjectToObject(lcd, "temporary_image");
        cJSON_AddBoolToObject(visible, "active", temporary.active);
        cJSON_AddNumberToObject(visible, "jpeg_bytes", temporary.jpeg_bytes);
        cJSON_AddNumberToObject(visible, "crc32", temporary.crc32);
        cJSON_AddNumberToObject(visible, "width", temporary.width);
        cJSON_AddNumberToObject(visible, "height", temporary.height);
    }
    cJSON *mqtt = cJSON_AddObjectToObject(root, "mqtt");
    cJSON_AddStringToObject(mqtt, "state", "disabled");
    cJSON *media = cJSON_AddObjectToObject(root, "media");
    cJSON_AddStringToObject(media, "mode", media_mode_name(media_mode));
    camera_status_t camera;
    media_mode_status(&camera);
    cJSON_AddNumberToObject(media, "captures", camera.completed);
    cJSON_AddNumberToObject(media, "failed_captures", camera.failed);
    cJSON_AddNumberToObject(media, "last_capture_ms", camera.last_capture_ms);
    cJSON_AddStringToObject(media, "last_error", camera.last_error == ESP_OK ? "" : esp_err_to_name(camera.last_error));
    cJSON *heap = cJSON_AddObjectToObject(root, "heap");
    cJSON_AddNumberToObject(heap, "free", esp_get_free_heap_size());
    cJSON_AddNumberToObject(heap, "spiram_free", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    cJSON_AddNumberToObject(heap, "dma_free", heap_caps_get_free_size(MALLOC_CAP_DMA));
    cJSON_AddNumberToObject(heap, "dma_largest_free", heap_caps_get_largest_free_block(MALLOC_CAP_DMA));
    cJSON_AddNumberToObject(heap, "internal_free", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    cJSON_AddNumberToObject(heap, "internal_largest_free", heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    cJSON_AddItemToObject(root,"voice",voice_json());
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
    bool persistent = !strcmp(req->path, "/lcd/default-image");
    esp_err_t err = persistent ? lcd_renderer_set_default_image(req->body, req->body_len) :
        lcd_renderer_push_image(req->body, req->body_len, duration_ms);
    if (err != ESP_OK) {
        int n = snprintf(resp, cap, "{\"ok\":false,\"error\":\"%s\"}", esp_err_to_name(err));
        if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
        return err == ESP_ERR_INVALID_ARG || err == ESP_ERR_INVALID_SIZE ||
            err == ESP_ERR_INVALID_RESPONSE ? 400 : 503;
    }
    int n = persistent ? snprintf(resp, cap, "{\"ok\":true,\"persistent\":true}") :
        snprintf(resp, cap, "{\"ok\":true,\"duration_ms\":%lu}", (unsigned long)duration_ms);
    if (resp_len) *resp_len = n > 0 ? (size_t)n : 0u;
    ESP_LOGI(TAG, "lcd_image: response_sent");
    return 202;
}

static int h_lcd_default_get(const nyx_proxy_request_t *req, char *resp, size_t cap,
                           char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    snprintf(ct, ct_cap, "application/json");
    lcd_default_image_info_t info = {0};
    esp_err_t err = lcd_renderer_default_image_info(&info);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 503;
    }
    snprintf(resp, cap, "{\"source\":\"%s\",\"persistent\":%s,\"width\":%u,\"height\":%u,\"jpeg_bytes\":%u}",
        info.stored ? "stored" : "builtin", info.stored ? "true" : "false",
        info.width, info.height, (unsigned)info.jpeg_bytes);
    json_response_len(resp, resp_len);
    return 200;
}

static int h_lcd_default_delete(const nyx_proxy_request_t *req, char *resp, size_t cap,
                              char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    snprintf(ct, ct_cap, "application/json");
    esp_err_t err = lcd_renderer_reset_default_image();
    snprintf(resp, cap, "{\"ok\":%s,\"err\":\"%s\"}", err == ESP_OK ? "true" : "false", esp_err_to_name(err));
    json_response_len(resp, resp_len);
    return err == ESP_OK ? 200 : 503;
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

// Compatibility routes cannot reactivate the retired continuous publisher.
static int h_stream_removed(const nyx_proxy_request_t *req, char *resp, size_t cap,
    char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req; (void)user;
    snprintf(ct, ct_cap, "application/json");
    snprintf(resp, cap, "{\"error\":\"continuous_streaming_removed\",\"replacement\":\"/camera/snapshot.jpg\"}");
    json_response_len(resp, resp_len);
    return 410;
}

static int h_stream_status(const nyx_proxy_request_t *req, char *resp, size_t cap,
    char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req; (void)user;
    snprintf(ct, ct_cap, "application/json");
    snprintf(resp, cap, "{\"state\":\"disabled\",\"auto_start\":false,\"target_whip_url\":\"\",\"last_error\":\"\"}");
    json_response_len(resp, resp_len);
    return 200;
}

static int h_stream_stop(const nyx_proxy_request_t *req, char *resp, size_t cap,
    char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req; (void)user;
    snprintf(ct, ct_cap, "application/json");
    snprintf(resp, cap, "{\"ok\":true,\"state\":\"disabled\"}");
    json_response_len(resp, resp_len);
    return 200;
}

static int h_camera_status(const nyx_proxy_request_t *req, char *resp, size_t cap,
    char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req; (void)user;
    camera_status_t state;
    media_mode_status(&state);
    snprintf(ct, ct_cap, "application/json");
    snprintf(resp, cap,
        "{\"mode\":\"%s\",\"width\":%u,\"height\":%u,\"content_type\":\"image/jpeg\","
        "\"snapshot_path\":\"/camera/snapshot.jpg\",\"continuous_streaming\":false,"
        "\"captures\":%lu,\"failed_captures\":%lu,\"last_capture_ms\":%lu,\"last_error\":\"%s\"}",
        media_mode_name(state.mode), PANEL_CAMERA_WIDTH, PANEL_CAMERA_HEIGHT,
        (unsigned long)state.completed, (unsigned long)state.failed,
        (unsigned long)state.last_capture_ms, state.last_error == ESP_OK ? "" : esp_err_to_name(state.last_error));
    json_response_len(resp, resp_len);
    return 200;
}

static int h_camera_openapi(const nyx_proxy_request_t *req, char *resp, size_t cap,
    char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;snprintf(ct,ct_cap,"application/json");
    cJSON *root=cJSON_CreateObject();cJSON_AddStringToObject(root,"openapi","3.0.3");
    cJSON *info=cJSON_AddObjectToObject(root,"info");
    cJSON_AddStringToObject(info,"title",!strcmp(req->service_slug,nyx_creds_service_slug())?"Nyx Panel Daily":"Nyx Panel Management");
    cJSON_AddStringToObject(info,"version",esp_app_get_description()->version);
    cJSON_AddStringToObject(info,"description","Daily actions and management settings are separately authorized NyxID services. Voice uses the board's own Key and selected Bot. Speech notifications never grant permission to tools.");
    cJSON *paths=cJSON_AddObjectToObject(root,"paths");
    size_t count=0;const nyx_route_t *routes=local_api_routes(&count);
    for(size_t i=0;i<count;++i){
        const nyx_route_t *r=&routes[i];
        if(!strcmp(r->path,"/openapi.json") || !panel_service_can_call(nyx_creds_service_slug(),req->service_slug,r->method,r->path))continue;
        cJSON *path=cJSON_GetObjectItemCaseSensitive(paths,r->path);
        if(!path)path=cJSON_AddObjectToObject(paths,r->path);
        char method[16],operation[128];snprintf(method,sizeof(method),"%s",r->method);
        for(char *p=method;*p;++p)if(*p>='A' && *p<='Z')*p+=32;
        snprintf(operation,sizeof(operation),"%s_%s",method,r->path+1);
        for(char *p=operation;*p;++p)if(*p=='/' || *p=='.' || *p=='-')*p='_';
        cJSON *op=cJSON_AddObjectToObject(path,method);
        cJSON_AddStringToObject(op,"operationId",operation);
        const char *description=r->path;
        if(!strcmp(r->path,"/voice/open"))description="Open the selected Bot in automatic two-way conversation and show voice controls. Optional text is a spoken remote notification. microphone_muted defaults false. Does not change the saved input mode. Reuse the same request_id to retry safely; last 8 remote requests survive reboot. A different call while busy returns 409.";
        else if(!strcmp(r->path,"/voice/say"))description="Queue a spoken notification in the active call. Not user input or tool authorization. Use a unique request_id; retries never repeat speech. accepted means provider acknowledgement, not proof of playback.";
        else if(!strcmp(r->path,"/voice/microphone"))description="Mute or unmute microphone input for the current call. Does not mute Bot audio.";
        else if(!strcmp(r->path,"/voice/stop"))description="End the current call and release microphone, speaker and WebRTC resources.";
        else if(!strcmp(r->path,"/voice/start"))description="Start a call with the saved input mode. Use /voice/open for remote automatic two-way conversation.";
        else if(!strcmp(r->path,"/voice") || !strcmp(r->path,"/status"))description="Read live device state including selected Bot, call state, microphone mute, media counters, remote request stage, resources and service scopes. No credentials are returned.";
        else if(!strcmp(r->path,"/voice/config"))description="Manage saved voice selection. POST exactly one of agent_id (from /voice/agents), input_mode (automatic or push_to_talk), playback_gain (integer 1..8). Selection and mode require idle. Persists in this server profile.";
        else if(!strcmp(r->path,"/config/audio"))description="Adjust saved audio output. POST exactly one of volume (integer 0..100) or playback_gain (integer 1..8). GET returns current audio and voice state. Does not change Bot selection.";
        else if(!strcmp(r->path,"/voice/input-mode"))description="Set saved input_mode to automatic or push_to_talk while idle. Exactly one field. Does not change the selected Bot or grants.";
        else if(!strcmp(r->path,"/voice/greeting"))description="Read or configure this panel's automatic spoken greeting for future calls. POST enabled (boolean). The selected Bot generates its greeting from its NyxID identity, persona and language; there is no scripted text. Persists in this server profile. An explicit remote opening message takes priority. Never a user utterance or tool permission.";
        else if(!strcmp(r->path,"/lcd/brightness"))description="Read or set saved screen brightness (integer 0..100). POST exactly one brightness field; preserves other display settings.";
        else if(!strcmp(r->path,"/lcd/default-image"))description="Manage the persistent JPEG fallback used when no temporary image is present. POST image/jpeg up to 65536 bytes. DELETE restores bundled NyxBot. GET returns metadata.";
        else if(!strcmp(r->path,"/lcd/image"))description="Show a temporary fullscreen JPEG, image/jpeg up to 65536 bytes; query duration_ms=0 is sticky until cleared, positive TTL returns to fallback.";
        else if(!strcmp(r->path,"/lcd/text"))description="Overlay text on the image. JSON text, size small|medium|large, color and bg #RRGGBB, align left|center|right, duration_ms. bg sets outline color.";
        else if(!strcmp(r->path,"/lcd/clear"))description="Clear temporary images and captions, keep the persistent fallback.";
        else if(!strcmp(r->path,"/system/update"))description="Manage stable firmware updates for this panel. GET reads version, automatic, available, state and blocked_version. POST exactly one field: automatic (boolean), or action (check or install). Asynchronous: 202 queues work; poll /status update and ota. Installation waits for idle. A rolled-back version is blocked. Management service only.";
        else if(!strcmp(r->path,"/camera/capture"))description="Take one camera photo, show the same photo fullscreen on Nyx Panel for two minutes, and return that JPEG for visual understanding. Keep voice controls and captions. Does not change the saved fallback image. No continuous stream.";
        else if(!strcmp(r->path,"/camera/snapshot.jpg") || !strcmp(r->path,"/snapshot.jpg"))description="Take a fresh native 800x1280 JPEG without changing the display. Use POST /camera/capture for a user-facing photo that is also shown on the panel. Release camera resources afterwards. Concurrent capture returns 503. No continuous stream.";
        else if(!strcmp(r->path,"/lcd/snapshot.jpg"))description="Read a JPEG screenshot of the current LCD.";
        cJSON_AddStringToObject(op,"summary",description);
        cJSON *responses=cJSON_AddObjectToObject(op,"responses");
        cJSON *ok=cJSON_AddObjectToObject(responses,"200");cJSON_AddStringToObject(ok,"description","Success");
        if(!strcmp(r->path,"/camera/capture") || !strcmp(r->path,"/camera/snapshot.jpg") ||
            !strcmp(r->path,"/snapshot.jpg") || !strcmp(r->path,"/lcd/snapshot.jpg")){
            cJSON *content=cJSON_AddObjectToObject(ok,"content");
            cJSON *media=cJSON_AddObjectToObject(content,"image/jpeg");
            cJSON_AddStringToObject(cJSON_AddObjectToObject(media,"schema"),"type","string");
        }
        cJSON *denied=cJSON_AddObjectToObject(responses,"403");cJSON_AddStringToObject(denied,"description","Service permission required");
        if(!strcmp(r->method,"POST")){
            cJSON *bad=cJSON_AddObjectToObject(responses,"400");cJSON_AddStringToObject(bad,"description","Invalid payload");
            cJSON *busy=cJSON_AddObjectToObject(responses,"409");cJSON_AddStringToObject(busy,"description","Conflict or busy");
            if(!strcmp(r->path,"/camera/capture"))continue;
            cJSON *body=cJSON_AddObjectToObject(op,"requestBody");
            cJSON *content=cJSON_AddObjectToObject(body,"content");
            bool jpeg=!strcmp(r->path,"/lcd/image") || !strcmp(r->path,"/lcd/default-image");
            cJSON *media=cJSON_AddObjectToObject(content,jpeg?"image/jpeg":"application/json");
            cJSON *schema=cJSON_AddObjectToObject(media,"schema");cJSON_AddStringToObject(schema,"type",jpeg?"string":"object");
            if(jpeg)cJSON_AddStringToObject(schema,"format","binary");
            else if(!strcmp(r->path,"/voice/open") || !strcmp(r->path,"/voice/say")){
                cJSON_AddBoolToObject(schema,"additionalProperties",false);
                cJSON *properties=cJSON_AddObjectToObject(schema,"properties");
                cJSON *id=cJSON_AddObjectToObject(properties,"request_id");cJSON_AddStringToObject(id,"type","string");cJSON_AddStringToObject(id,"format","uuid");
                cJSON *text=cJSON_AddObjectToObject(properties,"text");cJSON_AddStringToObject(text,"type","string");cJSON_AddStringToObject(text,"description","Nonblank UTF-8, max 300 bytes; no control characters.");
                cJSON *required=cJSON_AddArrayToObject(schema,"required");cJSON_AddItemToArray(required,cJSON_CreateString("request_id"));
                if(!strcmp(r->path,"/voice/say"))cJSON_AddItemToArray(required,cJSON_CreateString("text"));
                else{cJSON *mute=cJSON_AddObjectToObject(properties,"microphone_muted");cJSON_AddStringToObject(mute,"type","boolean");cJSON_AddBoolToObject(mute,"default",false);}
                cJSON *queued=cJSON_AddObjectToObject(responses,"202");cJSON_AddStringToObject(queued,"description","Queued asynchronously; inspect /voice for actual state");
            }else if(!strcmp(r->path,"/system/update")){
                cJSON_AddBoolToObject(schema,"additionalProperties",false);
                cJSON_AddNumberToObject(schema,"minProperties",1);
                cJSON_AddNumberToObject(schema,"maxProperties",1);
                cJSON *props=cJSON_AddObjectToObject(schema,"properties");
                cJSON_AddStringToObject(cJSON_AddObjectToObject(props,"automatic"),"type","boolean");
                cJSON *action=cJSON_AddObjectToObject(props,"action");cJSON_AddStringToObject(action,"type","string");
                cJSON *values=cJSON_AddArrayToObject(action,"enum");
                cJSON_AddItemToArray(values,cJSON_CreateString("check"));cJSON_AddItemToArray(values,cJSON_CreateString("install"));
                cJSON *queued=cJSON_AddObjectToObject(responses,"202");
                cJSON_AddStringToObject(queued,"description","Queued; poll /status update and ota for completion");
            }else if(!strcmp(r->path,"/voice/greeting")){
                cJSON_AddBoolToObject(schema,"additionalProperties",false);
                cJSON *required=cJSON_AddArrayToObject(schema,"required");
                cJSON_AddItemToArray(required,cJSON_CreateString("enabled"));
                cJSON *props=cJSON_AddObjectToObject(schema,"properties");
                cJSON_AddStringToObject(cJSON_AddObjectToObject(props,"enabled"),"type","boolean");
            }else if(!strcmp(r->path,"/voice/microphone")){
                cJSON_AddBoolToObject(schema,"additionalProperties",false);
                cJSON *props=cJSON_AddObjectToObject(schema,"properties");
                cJSON_AddStringToObject(cJSON_AddObjectToObject(props,"muted"),"type","boolean");
                cJSON_AddItemToArray(cJSON_AddArrayToObject(schema,"required"),cJSON_CreateString("muted"));
            }else if(!strcmp(r->path,"/voice/input-mode") || !strcmp(r->path,"/lcd/brightness") || !strcmp(r->path,"/config/audio")){
                cJSON_AddBoolToObject(schema,"additionalProperties",false);
                cJSON *props=cJSON_AddObjectToObject(schema,"properties");
                if(!strcmp(r->path,"/voice/input-mode")){
                    cJSON *mode=cJSON_AddObjectToObject(props,"input_mode");cJSON_AddStringToObject(mode,"type","string");
                    cJSON *values=cJSON_AddArrayToObject(mode,"enum");
                    cJSON_AddItemToArray(values,cJSON_CreateString("automatic"));cJSON_AddItemToArray(values,cJSON_CreateString("push_to_talk"));
                    cJSON_AddItemToArray(cJSON_AddArrayToObject(schema,"required"),cJSON_CreateString("input_mode"));
                }else if(!strcmp(r->path,"/lcd/brightness")){
                    cJSON_AddStringToObject(cJSON_AddObjectToObject(props,"brightness"),"type","integer");
                    cJSON_AddItemToArray(cJSON_AddArrayToObject(schema,"required"),cJSON_CreateString("brightness"));
                }else{
                    cJSON_AddStringToObject(cJSON_AddObjectToObject(props,"volume"),"type","integer");
                    cJSON_AddStringToObject(cJSON_AddObjectToObject(props,"playback_gain"),"type","integer");
                    cJSON_AddStringToObject(schema,"description","Exactly one field: volume 0..100 or playback_gain 1..8.");
                }
            }
        }
    }
    return object_response(root,resp,cap,resp_len);
}

static int h_snapshot(const nyx_proxy_request_t *req, char *resp, size_t cap,
                      char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    snprintf(ct, ct_cap, "image/jpeg");
    size_t len = 0;
    esp_err_t err = media_mode_snapshot((uint8_t *)resp, cap, &len, true);
    if (err != ESP_OK) {
        snprintf(ct, ct_cap, "application/json");
        snprintf(resp, cap, "{\"error\":\"camera_unavailable\",\"reason\":\"%s\"}", esp_err_to_name(err));
        json_response_len(resp, resp_len);
        return 503;
    }
    if (!strcmp(req->path, "/camera/capture")) {
        err = lcd_renderer_push_image((const uint8_t *)resp, len, 120000);
        if (err != ESP_OK) {
            snprintf(ct, ct_cap, "application/json");
            snprintf(resp, cap, "{\"error\":\"photo_captured_but_display_failed\",\"reason\":\"%s\"}", esp_err_to_name(err));
            json_response_len(resp, resp_len);
            return 503;
        }
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


static cJSON *update_json(void)
{
    panel_update_snapshot_t update;panel_update_snapshot(&update);
    cJSON *json=cJSON_CreateObject();
    cJSON_AddStringToObject(json,"current_version",esp_app_get_description()->version);
    cJSON_AddStringToObject(json,"profile",panel_update_profile());
    cJSON_AddBoolToObject(json,"automatic",update.automatic);
    cJSON_AddBoolToObject(json,"checking",update.checking);
    cJSON_AddBoolToObject(json,"available",update.available);
    cJSON_AddStringToObject(json,"latest_version",update.latest_version);
    cJSON_AddStringToObject(json,"blocked_version",update.blocked_version);
    cJSON_AddStringToObject(json,"state",update.state);
    cJSON_AddStringToObject(json,"last_error",update.last_error==ESP_OK ? "" : esp_err_to_name(update.last_error));
    return json;
}
static int h_system_update(const nyx_proxy_request_t *req,char *resp,size_t cap,
                           char *ct,size_t ct_cap,size_t *resp_len,void *user)
{
    (void)user;snprintf(ct,ct_cap,"application/json");
    if (!strcmp(req->method,"GET")) return object_response(update_json(),resp,cap,resp_len);
    int status=0;cJSON *root=parse_json_body(req,resp,cap,resp_len,&status);
    if (!root) return status;
    cJSON *enabled=cJSON_GetObjectItemCaseSensitive(root,"automatic");
    cJSON *action=cJSON_GetObjectItemCaseSensitive(root,"action");
    esp_err_t err=ESP_ERR_INVALID_ARG;
    if (cJSON_GetArraySize(root)==1) {
        if (cJSON_IsBool(enabled)) err=panel_update_set_automatic(cJSON_IsTrue(enabled));
        else if (cJSON_IsString(action) && (!strcmp(action->valuestring,"check") || !strcmp(action->valuestring,"install")))
            err=panel_update_check(!strcmp(action->valuestring,"install"));
    }
    cJSON_Delete(root);
    if (err!=ESP_OK) {
        snprintf(resp,cap,"{\"error\":\"%s\"}",esp_err_to_name(err));json_response_len(resp,resp_len);
        return err==ESP_ERR_INVALID_ARG ? 400 : err==ESP_ERR_INVALID_STATE ? 409 : 503;
    }
    int result=object_response(update_json(),resp,cap,resp_len);
    return result==200 ? 202 : result;
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
    esp_err_t err = panel_ota_start_url(url, expected, NULL, 0);
    free(url);
    if (err != ESP_OK) {
        snprintf(resp, cap, "{\"error\":\"ota_unavailable\"}");
        json_response_len(resp, resp_len);
        return err == ESP_ERR_INVALID_STATE ? 409 : 503;
    }
    // Nyx proxies have a finite response deadline. Download in a separate worker;
    // 202 acknowledges acceptance, and /status reports completion or failure.
    snprintf(resp, cap, "{\"ok\":true,\"accepted\":true,\"ota\":true}");
    json_response_len(resp, resp_len);
    return 202;
}

static int h_nyx_login_status(const nyx_proxy_request_t *req, char *resp, size_t cap,
                              char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)req; (void)user;
    snprintf(ct, ct_cap, "application/json");
    device_login_snapshot_t state;
    device_code_snapshot(&state);
    static const char *names[] = {"idle", "requesting", "waiting", "registering", "service", "ready", "cancelled", "failed"};
    cJSON *json = cJSON_CreateObject();
    if (!json) return 503;
    cJSON_AddStringToObject(json, "state", names[state.state]);
    cJSON_AddBoolToObject(json, "busy", state.busy);
    cJSON_AddStringToObject(json, "login_kind", state.node_setup ? "remote_control_setup" : "agent_key");
    cJSON_AddStringToObject(json, "message", state.message);
    cJSON_AddStringToObject(json, "user_code", state.user_code);
    cJSON_AddStringToObject(json, "verification_uri", state.verification_uri);
    cJSON_AddStringToObject(json, "verification_uri_complete", state.verification_uri_complete);
    cJSON_AddNumberToObject(json, "seconds_remaining", state.seconds_remaining);
    return object_response(json, resp, cap, resp_len);
}

static int h_nyx_login_start(const nyx_proxy_request_t *req, char *resp, size_t cap,
                             char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    esp_err_t err = lcd_renderer_navigate("device_bind");
    if (err == ESP_OK) err = device_code_start_if_needed();
    if (err != ESP_OK) {
        snprintf(ct, ct_cap, "application/json");
        snprintf(resp, cap, "{\"error\":\"network_unavailable_or_sign_in_busy\"}");
        json_response_len(resp, resp_len);
        return 409;
    }
    (void)h_nyx_login_status(req, resp, cap, ct, ct_cap, resp_len, NULL);
    return 202;
}

static int h_nyx_login_cancel(const nyx_proxy_request_t *req, char *resp, size_t cap,
                              char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    device_code_cancel();
    (void)h_nyx_login_status(req, resp, cap, ct, ct_cap, resp_len, user);
    return 202;
}

static int h_wifi_qr(const nyx_proxy_request_t *req, char *resp, size_t cap,
                      char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;
    if (!strcmp(req->method, "POST")) qr_scanner_confirm();
    qr_scan_snapshot_t state;
    qr_scanner_snapshot(&state);
    static const char *names[] = {"scanning", "confirm", "connecting", "connected", "error"};
    snprintf(ct, ct_cap, "application/json");
    cJSON *json = cJSON_CreateObject();
    if (!json) return 503;
    cJSON_AddStringToObject(json, "state", names[state.state]);
    cJSON_AddStringToObject(json, "ssid", state.ssid);
    cJSON_AddStringToObject(json, "message", state.message);
    return object_response(json, resp, cap, resp_len);
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

    qr_scan_snapshot_t scan;
    qr_scanner_snapshot(&scan);
    snprintf(resp, cap, "{\"ok\":true,\"confirmation_required\":%s}",
        scan.state == QR_SCAN_CONFIRM ? "true" : "false");
    json_response_len(resp, resp_len);
    return 202;
}
#endif

// Strict small JSON objects for configuration/control: no trailing input, NULs or duplicate keys.
static cJSON *control_object(const nyx_proxy_request_t *req)
{
    if(!req->body || !req->body_len || req->body_len>1024 || memchr(req->body,0,req->body_len))return NULL;
    for(size_t i=0;i+6<=req->body_len;++i)if(!memcmp(req->body+i,"\\u0000",6))return NULL;
    const char *end=NULL,*raw=(const char *)req->body;
    cJSON *j=cJSON_ParseWithLengthOpts(raw,req->body_len,&end,false);
    if(end)while((size_t)(end-raw)<req->body_len && strchr(" \t\r\n",*end))++end;
    bool ok=cJSON_IsObject(j) && end && (size_t)(end-raw)==req->body_len;
    const cJSON *x,*y;
    cJSON_ArrayForEach(x,j)for(y=x->next;y;y=y->next)if(!strcmp(x->string,y->string))ok=false;
    if(!ok){cJSON_Delete(j);return NULL;}return j;
}
static bool uint_field(const cJSON *v,unsigned low,unsigned high)
{
    return cJSON_IsNumber(v) && v->valuedouble>=low && v->valuedouble<=high && floor(v->valuedouble)==v->valuedouble;
}
static int h_brightness(const nyx_proxy_request_t *req, char *resp, size_t cap,
                        char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;snprintf(ct,ct_cap,"application/json");
    cJSON *j=NULL,*value=NULL;
    bool update=!strcmp(req->method,"POST");
    if(update){
        j=control_object(req);value=cJSON_GetObjectItemCaseSensitive(j,"brightness");
        if(cJSON_GetArraySize(j)!=1 || !uint_field(value,0,100)){
            cJSON_Delete(j);snprintf(resp,cap,"{\"error\":\"brightness_must_be_integer_0_to_100\"}");
            json_response_len(resp,resp_len);return 400;
        }
    }
    cfg_lcd_t cfg={0};
    esp_err_t err=nvs_config_get_lcd(&cfg);
    if(err==ESP_OK && update){
        cfg.brightness=(uint8_t)value->valueint;
        err=nvs_config_set_lcd(&cfg);
        if(err==ESP_OK){(void)bsp_display_brightness_set(cfg.brightness);lcd_renderer_reload_config();}
    }
    cJSON_Delete(j);
    if(err!=ESP_OK){
        snprintf(resp,cap,"{\"error\":\"%s\"}",esp_err_to_name(err));json_response_len(resp,resp_len);return 500;
    }
    cJSON *result=cJSON_CreateObject();cJSON_AddNumberToObject(result,"brightness",cfg.brightness);
    return object_response(result,resp,cap,resp_len);
}
static int h_voice_greeting(const nyx_proxy_request_t *req, char *resp, size_t cap,
                  char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;snprintf(ct,ct_cap,"application/json");
    if(!strcmp(req->method,"POST")) {
        esp_err_t err=panel_voice_greeting_update(req->body,req->body_len);
        if(err!=ESP_OK) {
            snprintf(resp,cap,"{\"error\":\"%s\"}",err==ESP_ERR_INVALID_ARG?
                "expected_only_enabled_boolean":esp_err_to_name(err));
            json_response_len(resp,resp_len);return err==ESP_ERR_INVALID_ARG?400:503;
        }
    }
    panel_voice_greeting_t settings;panel_voice_greeting_get(&settings);
    cJSON *j=cJSON_CreateObject();cJSON_AddBoolToObject(j,"enabled",settings.enabled);
    return object_response(j,resp,cap,resp_len);
}
static int h_voice(const nyx_proxy_request_t *req, char *resp, size_t cap,
                  char *ct, size_t ct_cap, size_t *resp_len, void *user)
{
    (void)user;snprintf(ct,ct_cap,"application/json");
    esp_err_t err=ESP_OK;bool duplicate=false,remote=false;
    panel_voice_remote_t request={0};
    if(!strcmp(req->path,"/voice/open") || !strcmp(req->path,"/voice/say")){
        remote=true;bool open=!strcmp(req->path,"/voice/open");
        if(!panel_voice_parse_remote(req->body,req->body_len,open,&request))err=ESP_ERR_INVALID_ARG;
        else err=open?panel_voice_remote_open(&request,&duplicate):panel_voice_remote_say(&request,&duplicate);
    }else if(!strcmp(req->path,"/voice/microphone")){
        cJSON *j=control_object(req),*v=cJSON_GetObjectItemCaseSensitive(j,"muted");
        if(cJSON_GetArraySize(j)!=1 || !cJSON_IsBool(v))err=ESP_ERR_INVALID_ARG;
        else panel_voice_microphone_mute(cJSON_IsTrue(v));
        cJSON_Delete(j);
    }else if((!strcmp(req->path,"/voice/config") || !strcmp(req->path,"/config/audio") || !strcmp(req->path,"/voice/input-mode")) && !strcmp(req->method,"POST")){
        cJSON *j=control_object(req),*v=j?j->child:NULL;
        if(cJSON_GetArraySize(j)!=1 || !v)err=ESP_ERR_INVALID_ARG;
        else if(!strcmp(req->path,"/voice/config") && !strcmp(v->string,"agent_id") && cJSON_IsString(v))err=panel_voice_configure(v->valuestring);
        else if((!strcmp(req->path,"/voice/config") || !strcmp(req->path,"/voice/input-mode")) && !strcmp(v->string,"input_mode") && cJSON_IsString(v)){
            if(!strcmp(v->valuestring,"automatic"))err=panel_voice_input_mode_set(PANEL_VOICE_INPUT_AUTOMATIC);
            else if(!strcmp(v->valuestring,"push_to_talk"))err=panel_voice_input_mode_set(PANEL_VOICE_INPUT_HOLD);
            else err=ESP_ERR_INVALID_ARG;
        }else if(strcmp(req->path,"/voice/input-mode") && !strcmp(v->string,"playback_gain") && uint_field(v,1,8))err=panel_voice_gain_set((unsigned)v->valueint);
        else if(!strcmp(req->path,"/config/audio") && !strcmp(v->string,"volume") && uint_field(v,0,100))err=panel_voice_volume_set((unsigned)v->valueint);
        else err=ESP_ERR_INVALID_ARG;
        cJSON_Delete(j);
    }else if(!strcmp(req->path,"/voice/agents/refresh"))err=panel_voice_refresh_agents();
    else if(!strcmp(req->path,"/voice/agents")){
        panel_voice_bot_t *bots=calloc(PANEL_VOICE_MAX_BOTS,sizeof(*bots));
        if(!bots)return 503;
        size_t count=panel_voice_bots(bots,PANEL_VOICE_MAX_BOTS);
        cJSON *root=cJSON_CreateObject(),*rows=cJSON_AddArrayToObject(root,"agents");
        for(size_t i=0;i<count;++i){
            cJSON *item=cJSON_CreateObject();cJSON_AddStringToObject(item,"agent_id",bots[i].agent_id);
            cJSON_AddStringToObject(item,"service_id",bots[i].service_id);
            cJSON_AddStringToObject(item,"name",bots[i].name);cJSON_AddItemToArray(rows,item);
        }
        free(bots);return object_response(root,resp,cap,resp_len);
    }else if(!strcmp(req->path,"/voice/start")){
        err=panel_voice_start();if(err==ESP_OK)(void)lcd_renderer_show_voice();
    }else if(!strcmp(req->path,"/voice/stop"))panel_voice_stop();
    else if(!strcmp(req->path,"/voice/hold"))panel_voice_hold(query_u32(req->query,"pressed",0)!=0);
    else if(!strcmp(req->path,"/voice/peer-test")){
        err=panel_voice_peer_test(resp,cap);if(err==ESP_OK){json_response_len(resp,resp_len);return 200;}
    }else if(!strcmp(req->path,"/voice/test")){
        err=panel_voice_test(resp,cap);if(err==ESP_OK){json_response_len(resp,resp_len);return 200;}
    }
    if(err!=ESP_OK){
        snprintf(resp,cap,"{\"ok\":false,\"error\":\"%s\"}",esp_err_to_name(err));json_response_len(resp,resp_len);
        return err==ESP_ERR_INVALID_ARG?400:err==ESP_ERR_NO_MEM?503:409;
    }
    cJSON *j=voice_json();
    if(remote){cJSON_AddStringToObject(j,"request_id",request.request_id);cJSON_AddBoolToObject(j,"duplicate",duplicate);}
    int result=object_response(j,resp,cap,resp_len);
    return result==200 && remote && !duplicate?202:result;
}

static const nyx_route_t s_routes[] = {
    { .path = "/lcd/brightness", .method = "GET", .fn = h_brightness },
    { .path = "/lcd/brightness", .method = "POST", .fn = h_brightness },
    { .path = "/voice/input-mode", .method = "POST", .fn = h_voice },
    { .path = "/voice", .method = "GET", .fn = h_voice },
    { .path = "/voice/config", .method = "POST", .fn = h_voice },
    { .path = "/voice/config", .method = "GET", .fn = h_voice },
    { .path = "/voice/greeting", .method = "GET", .fn = h_voice_greeting },
    { .path = "/voice/greeting", .method = "POST", .fn = h_voice_greeting },
    { .path = "/config/audio", .method = "GET", .fn = h_voice },
    { .path = "/config/audio", .method = "POST", .fn = h_voice },
    { .path = "/voice/open", .method = "POST", .fn = h_voice },
    { .path = "/voice/say", .method = "POST", .fn = h_voice },
    { .path = "/voice/microphone", .method = "POST", .fn = h_voice },
    { .path = "/voice/agents", .method = "GET", .fn = h_voice },
    { .path = "/voice/agents/refresh", .method = "POST", .fn = h_voice },
    { .path = "/voice/start", .method = "POST", .fn = h_voice },
    { .path = "/voice/stop", .method = "POST", .fn = h_voice },
    { .path = "/voice/hold", .method = "POST", .fn = h_voice },
    { .path = "/voice/test", .method = "POST", .fn = h_voice },
    { .path = "/voice/peer-test", .method = "POST", .fn = h_voice },
    { .path = "/status", .method = "GET", .fn = h_status, .user = NULL },
    { .path = "/nyx/login", .method = "GET", .fn = h_nyx_login_status, .user = NULL },
    { .path = "/nyx/login", .method = "POST", .fn = h_nyx_login_start, .user = NULL },
    { .path = "/nyx/login/cancel", .method = "POST", .fn = h_nyx_login_cancel, .user = NULL },
    { .path = "/wifi/qr", .method = "GET", .fn = h_wifi_qr, .user = NULL },
    { .path = "/wifi/qr", .method = "POST", .fn = h_wifi_qr, .user = NULL },
    { .path = "/lcd/image", .method = "POST", .fn = h_lcd_image, .user = NULL },
    { .path = "/lcd/default-image", .method = "GET", .fn = h_lcd_default_get, .user = NULL },
    { .path = "/lcd/default-image", .method = "POST", .fn = h_lcd_image, .user = NULL },
    { .path = "/lcd/default-image", .method = "DELETE", .fn = h_lcd_default_delete, .user = NULL },
    { .path = "/lcd/text", .method = "POST", .fn = h_lcd_text, .user = NULL },
    { .path = "/lcd/clear", .method = "POST", .fn = h_lcd_clear, .user = NULL },
    { .path = "/lcd/snapshot.jpg", .method = "GET", .fn = h_lcd_snapshot, .user = NULL },
    { .path = "/snapshot.jpg", .method = "GET", .fn = h_snapshot, .user = NULL },
    { .path = "/camera/snapshot.jpg", .method = "GET", .fn = h_snapshot, .user = NULL },
    { .path = "/camera/capture", .method = "POST", .fn = h_snapshot, .user = NULL },
    { .path = "/camera", .method = "GET", .fn = h_camera_status, .user = NULL },
    { .path = "/openapi.json", .method = "GET", .fn = h_camera_openapi, .user = NULL },
    { .path = "/stream/start", .method = "POST", .fn = h_stream_removed, .user = NULL },
    { .path = "/stream/stop", .method = "POST", .fn = h_stream_stop, .user = NULL },
    { .path = "/stream/status", .method = "GET", .fn = h_stream_status, .user = NULL },
    { .path = "/lcd/screen", .method = "GET", .fn = h_lcd_screen_get, .user = NULL },
    { .path = "/lcd/screen", .method = "POST", .fn = h_lcd_screen_post, .user = NULL },
    { .path = "/config/stream", .method = "GET", .fn = h_stream_status, .user = NULL },
    { .path = "/config/stream", .method = "POST", .fn = h_stream_removed, .user = NULL },
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
    { .path = "/system/update", .method = "GET", .fn = h_system_update, .user = NULL },
    { .path = "/system/update", .method = "POST", .fn = h_system_update, .user = NULL },
    { .path = "/system/ota", .method = "POST", .fn = h_system_ota, .user = NULL },
#if HA_CAM_DEV_QR_INJECT
    { .path = "/_dev/qr/inject_jpeg", .method = "POST", .fn = h_dev_inject_qr_jpeg, .user = NULL },
#endif
};

static const nyx_route_t s_lan_routes[] = {
    { .path = "/lcd/image", .method = "POST", .fn = h_lcd_image, .user = NULL },
    { .path = "/lcd/clear", .method = "POST", .fn = h_lcd_clear, .user = NULL },
    { .path = "/stream/start", .method = "POST", .fn = h_stream_removed, .user = NULL },
    { .path = "/stream/stop", .method = "POST", .fn = h_stream_stop, .user = NULL },
    { .path = "/stream/status", .method = "GET", .fn = h_stream_status, .user = NULL },
    { .path = "/config/stream", .method = "GET", .fn = h_stream_status, .user = NULL },
    { .path = "/config/stream", .method = "POST", .fn = h_stream_removed, .user = NULL },
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
    if (!req || !panel_service_can_call(nyx_creds_service_slug(),req->service_slug,req->method,req->path)) {
        snprintf(content_type,ct_cap,"application/json");
        snprintf(resp_body,resp_cap,"{\"error\":\"management_service_required_or_unknown_service\"}");
        json_response_len(resp_body,resp_len);
        return log_route_result(req,403,content_type,resp_len?*resp_len:0u);
    }
    if (!route || !route->fn) {
        return log_route_result(req, 404, content_type, resp_len ? *resp_len : 0u);
    }
    int status = route->fn(req, resp_body, resp_cap, content_type, ct_cap, resp_len, route->user);
    return log_route_result(req, status, content_type, resp_len ? *resp_len : 0u);
}

int local_api_lan_route_with_logging(const nyx_route_t *route,const nyx_proxy_request_t *req,
    char *body,size_t cap,char *ct,size_t ct_cap,size_t *len)
{
    // LAN has its own fixed, limited recovery routes. It cannot select a service scope.
    bool allowed=false;
    for(size_t i=0;i<sizeof(s_lan_routes)/sizeof(*s_lan_routes);++i)if(route==&s_lan_routes[i])allowed=true;
    int code=allowed && route->fn?route->fn(req,body,cap,ct,ct_cap,len,route->user):404;
    return log_route_result(req,code,ct,len?*len:0u);
}
