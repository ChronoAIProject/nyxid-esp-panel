#include "local_http_server.h"

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "heap_diag.h"
#include "lwip/tcp.h"
#include "lcd_renderer.h"
#include "lwip/sockets.h"
#include "local_api.h"
#include "media_mode.h"
#include "nvs_config.h"
#include "snapshot.h"
#include "common.h"
#include "panel_ota.h"
#include "nyx_creds.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "mbedtls/sha256.h"

#include <stdlib.h>
#include <ctype.h>
#include <sys/param.h>
#include "freertos/task.h"
#include <string.h>
#include <strings.h>

static const char *TAG = "haCam.lan";

#define LCD_IMAGE_MAX_BYTES (64u * 1024u)
#define LOCAL_API_DEFAULT_RESP_CAP (768u * 1024u)
#define LOCAL_API_FALLBACK_RESP_CAP (32u * 1024u)
#define LOCAL_API_SMALL_RESP_CAP 1024u
#define BODY_RECV_TIMEOUT_MS 30000u
#define BODY_RECV_CHUNK_BYTES 4096u

static httpd_handle_t s_server;
static SemaphoreHandle_t s_snapshot_lock;
static SemaphoreHandle_t s_lcd_snapshot_lock;

static const char *remote_ip(httpd_req_t *req, char *buf, size_t cap)
{
    int fd = httpd_req_to_sockfd(req);
    struct sockaddr_storage addr = {0};
    socklen_t addr_len = sizeof(addr);
    if (fd < 0 || getpeername(fd, (struct sockaddr *)&addr, &addr_len) != 0) {
        snprintf(buf, cap, "?");
        return buf;
    }
    if (addr.ss_family == AF_INET) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)&addr;
        uint32_t ip = ntohl(in->sin_addr.s_addr);
        snprintf(buf, cap, "%lu.%lu.%lu.%lu",
                 (unsigned long)((ip >> 24) & 0xffu),
                 (unsigned long)((ip >> 16) & 0xffu),
                 (unsigned long)((ip >> 8) & 0xffu),
                 (unsigned long)(ip & 0xffu));
    } else {
        snprintf(buf, cap, "?");
    }
    return buf;
}

static void log_request(httpd_req_t *req, int status, size_t body_len)
{
    char ip[24];
    ESP_LOGI(TAG, "request remote=%s method=%d path=%s status=%d body_len=%u",
             remote_ip(req, ip, sizeof(ip)), (int)req->method, req->uri,
             status, (unsigned)body_len);
    heap_diag_log(TAG, "http_request");
}

static size_t local_api_response_cap(const char *path)
{
    if (strcmp(path, "/lcd/image") == 0 ||
        strcmp(path, "/lcd/clear") == 0 ||
        strcmp(path, "/lcd/text") == 0) {
        return LOCAL_API_SMALL_RESP_CAP;
    }
    return LOCAL_API_DEFAULT_RESP_CAP;
}

static esp_err_t root_get(httpd_req_t *req)
{
    char status[4096];
    (void)local_api_format_status_json(status, sizeof(status));
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    esp_err_t err = httpd_resp_sendstr_chunk(req,
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>ESP32 Cam</title></head><body>"
        "<h1>ESP32 Cam</h1>"
        "<p><img src=\"/snapshot.jpg\" style=\"max-width:100%;height:auto\"></p>"
        "<h2>Status</h2><pre>");
    if (err == ESP_OK) {
        err = httpd_resp_sendstr_chunk(req, status);
    }
    if (err == ESP_OK) {
        err = httpd_resp_sendstr_chunk(req, "</pre></body></html>");
    }
    if (err == ESP_OK) {
        err = httpd_resp_sendstr_chunk(req, NULL);
    }
    log_request(req, err == ESP_OK ? 200 : 500, 0u);
    return err;
}

static esp_err_t status_get(httpd_req_t *req)
{
    char body[4096];
    size_t len = local_api_format_status_json(body, sizeof(body));
    httpd_resp_set_type(req, "application/json");
    esp_err_t err = httpd_resp_send(req, body, len);
    log_request(req, err == ESP_OK ? 200 : 500, len);
    return err;
}

static esp_err_t snapshot_get(httpd_req_t *req)
{
    if (!media_mode_is_preview()) {
        httpd_resp_send_custom_err(req, "503 Service Unavailable",
                                   "snapshot unavailable in publish mode");
        log_request(req, 503, 0u);
        return ESP_OK;
    }
    if (!s_snapshot_lock || xSemaphoreTake(s_snapshot_lock, pdMS_TO_TICKS(3000)) != pdTRUE) {
        httpd_resp_send_custom_err(req, "503 Service Unavailable", "snapshot busy");
        log_request(req, 503, 0u);
        return ESP_OK;
    }

    size_t grabbed_len = 0u;
    esp_err_t grab_err = media_sys_preview_snapshot_encode(2500u, &grabbed_len);
    if (grab_err == ESP_ERR_NOT_SUPPORTED || grab_err == ESP_ERR_INVALID_STATE) {
        grab_err = media_sys_snapshot_grab(2500u, &grabbed_len);
    }
    if (grab_err != ESP_OK) {
        xSemaphoreGive(s_snapshot_lock);
        httpd_resp_send_custom_err(req, "503 Service Unavailable", esp_err_to_name(grab_err));
        log_request(req, 503, 0u);
        return ESP_OK;
    }

    uint8_t *buf = heap_caps_malloc(SNAPSHOT_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        buf = heap_caps_malloc(SNAPSHOT_MAX_BYTES, MALLOC_CAP_8BIT);
    }
    if (!buf) {
        xSemaphoreGive(s_snapshot_lock);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "snapshot alloc failed");
        log_request(req, 500, 0u);
        return ESP_OK;
    }
    size_t len = 0u;
    esp_err_t copy_err = snapshot_copy_latest(buf, SNAPSHOT_MAX_BYTES, &len);
    if (copy_err != ESP_OK) {
        free(buf);
        xSemaphoreGive(s_snapshot_lock);
        httpd_resp_send_custom_err(req, "503 Service Unavailable", esp_err_to_name(copy_err));
        log_request(req, 503, 0u);
        return ESP_OK;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");
    esp_err_t err = httpd_resp_send(req, (const char *)buf, len);
    free(buf);
    xSemaphoreGive(s_snapshot_lock);
    (void)grabbed_len;
    log_request(req, err == ESP_OK ? 200 : 500, len);
    return err;
}

static esp_err_t lcd_snapshot_get(httpd_req_t *req)
{
    // Mutex with a generous wait window — LCD snapshot is expensive (~150ms
    // for a full framebuffer encode) so concurrent calls block instead of
    // immediately 503'ing. Tests can hit this back-to-back.
    if (!s_lcd_snapshot_lock || xSemaphoreTake(s_lcd_snapshot_lock, pdMS_TO_TICKS(3000)) != pdTRUE) {
        httpd_resp_send_custom_err(req, "503 Service Unavailable", "lcd snapshot busy");
        log_request(req, 503, 0u);
        return ESP_OK;
    }

    // 800x1280 RGB565 → ~512 KiB JPEG worst case at q=70; 768 KiB caps it safely.
    const size_t cap = 768u * 1024u;
    uint8_t *buf = heap_caps_malloc(cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "lcd snapshot alloc failed");
        log_request(req, 500, 0u);
        xSemaphoreGive(s_lcd_snapshot_lock);
        return ESP_OK;
    }
    size_t len = 0u;
    esp_err_t err = lcd_renderer_snapshot_jpeg(buf, cap, &len);
    if (err != ESP_OK) {
        free(buf);
        if (err == ESP_ERR_INVALID_STATE || err == ESP_ERR_TIMEOUT) {
            httpd_resp_send_custom_err(req, "503 Service Unavailable", esp_err_to_name(err));
            log_request(req, 503, 0u);
        } else {
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, esp_err_to_name(err));
            log_request(req, 500, 0u);
        }
        xSemaphoreGive(s_lcd_snapshot_lock);
        return ESP_OK;
    }
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");
    esp_err_t send_err = httpd_resp_send(req, (const char *)buf, len);
    free(buf);
    log_request(req, send_err == ESP_OK ? 200 : 500, len);
    xSemaphoreGive(s_lcd_snapshot_lock);
    return send_err;
}

// Extract `target=<value>` from a URL query string. Returns false if the
// param is missing, empty, or longer than out_cap-1.
static bool __attribute__((unused)) query_get_target(httpd_req_t *req, char *out, size_t out_cap)
{
    size_t qlen = httpd_req_get_url_query_len(req);
    if (qlen == 0u || qlen >= 256u) return false;
    char query[256];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) return false;
    return httpd_query_key_value(query, "target", out, out_cap) == ESP_OK
           && out[0] != '\0';
}

static esp_err_t lcd_screen_get(httpd_req_t *req)
{
    const char *target = "unknown";
    size_t depth = 0u;
    esp_err_t err = lcd_renderer_current_screen(&target, &depth);
    char body[96];
    int n = snprintf(body, sizeof(body),
                     "{\"target\":\"%s\",\"depth\":%u,\"err\":\"%s\"}",
                     target ? target : "unknown", (unsigned)depth,
                     esp_err_to_name(err));
    httpd_resp_set_type(req, "application/json");
    esp_err_t send_err = httpd_resp_send(req, body, n);
    log_request(req, send_err == ESP_OK ? 200 : 500, (size_t)n);
    return send_err;
}

static esp_err_t __attribute__((unused)) lcd_screen_post(httpd_req_t *req)
{
    char target[32] = {0};
    if (!query_get_target(req, target, sizeof(target))) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "missing or invalid ?target=<screen>");
        log_request(req, 400, 0u);
        return ESP_OK;
    }

    esp_err_t err = lcd_renderer_navigate(target);
    if (err != ESP_OK) {
        char body[128];
        int n = snprintf(body, sizeof(body),
                         "{\"ok\":false,\"target\":\"%s\",\"err\":\"%s\"}",
                         target, esp_err_to_name(err));
        httpd_resp_set_status(req, err == ESP_ERR_NOT_FOUND ? "404 Not Found"
                                                            : "500 Internal Server Error");
        httpd_resp_set_type(req, "application/json");
        httpd_resp_send(req, body, n);
        log_request(req, err == ESP_ERR_NOT_FOUND ? 404 : 500, (size_t)n);
        return ESP_OK;
    }

    char body[96];
    int n = snprintf(body, sizeof(body), "{\"ok\":true,\"target\":\"%s\"}", target);
    httpd_resp_set_type(req, "application/json");
    esp_err_t send_err = httpd_resp_send(req, body, n);
    log_request(req, send_err == ESP_OK ? 200 : 500, (size_t)n);
    return send_err;
}

static bool split_uri(httpd_req_t *req, char *path, size_t path_cap,
                      char *query, size_t query_cap)
{
    if (!req || !path || path_cap == 0u || !query || query_cap == 0u) {
        return false;
    }
    const char *q = strchr(req->uri, '?');
    size_t path_len = q ? (size_t)(q - req->uri) : strlen(req->uri);
    if (path_len == 0u || path_len >= path_cap) {
        return false;
    }
    memcpy(path, req->uri, path_len);
    path[path_len] = '\0';
    query[0] = '\0';
    if (q && q[1] != '\0') {
        size_t query_len = strlen(q + 1);
        if (query_len >= query_cap) {
            return false;
        }
        memcpy(query, q + 1, query_len + 1u);
    }
    return true;
}

static const char *method_name(httpd_method_t method)
{
    switch (method) {
    case HTTP_GET: return "GET";
    case HTTP_POST: return "POST";
    default: return "";
    }
}

static const nyx_route_t *find_local_api_route(const char *path, const char *method)
{
    size_t count = 0u;
    const nyx_route_t *routes = local_api_lan_routes(&count);
    for (size_t i = 0u; i < count; ++i) {
        if (routes[i].path && routes[i].method &&
            strcmp(routes[i].path, path) == 0 &&
            strcasecmp(routes[i].method, method) == 0) {
            return &routes[i];
        }
    }
    return NULL;
}

static bool request_expects_continue(httpd_req_t *req)
{
    if (!req) {
        return false;
    }
    size_t len = httpd_req_get_hdr_value_len(req, "Expect");
    if (len == 0u || len >= 32u) {
        return false;
    }
    char expect[32] = {0};
    if (httpd_req_get_hdr_value_str(req, "Expect", expect, sizeof(expect)) != ESP_OK) {
        return false;
    }
    return strcasecmp(expect, "100-continue") == 0;
}

static esp_err_t read_request_body(httpd_req_t *req, uint8_t **out, size_t *out_len)
{
    if (!req || !out || !out_len) {
        return ESP_ERR_INVALID_ARG;
    }
    *out = NULL;
    *out_len = 0u;
    if (req->content_len <= 0) {
        return ESP_OK;
    }
    uint8_t *body = heap_caps_malloc((size_t)req->content_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!body) {
        body = heap_caps_malloc((size_t)req->content_len, MALLOC_CAP_8BIT);
    }
    if (!body) {
        return ESP_ERR_NO_MEM;
    }
    if (request_expects_continue(req)) {
        static const char continue_resp[] = "HTTP/1.1 100 Continue\r\n\r\n";
        if (httpd_send(req, continue_resp, sizeof(continue_resp) - 1u) < 0) {
            free(body);
            return ESP_FAIL;
        }
    }
    size_t received = 0u;
    while (received < (size_t)req->content_len) {
        size_t remaining = (size_t)req->content_len - received;
        size_t chunk = remaining > BODY_RECV_CHUNK_BYTES ? BODY_RECV_CHUNK_BYTES : remaining;
        int64_t deadline_us = esp_timer_get_time() + (int64_t)BODY_RECV_TIMEOUT_MS * 1000;
        int rc = httpd_req_recv(req, (char *)body + received,
                                chunk);
        while (rc == HTTPD_SOCK_ERR_TIMEOUT && esp_timer_get_time() < deadline_us) {
            vTaskDelay(pdMS_TO_TICKS(10));
            rc = httpd_req_recv(req, (char *)body + received, chunk);
        }
        if (rc <= 0) {
            free(body);
            return ESP_FAIL;
        }
        received += (size_t)rc;
    }
    *out = body;
    *out_len = received;
    return ESP_OK;
}

static esp_err_t reject_lcd_image_too_large(httpd_req_t *req)
{
    int fd = httpd_req_to_sockfd(req);
    char body[224];
    int n = snprintf(body, sizeof(body),
        "{\"error\":\"payload_too_large\",\"max_bytes\":%u,"
        "\"received\":%lu,\"hint\":\"resize+recompress before POST; "
        "see HA service template in tools/ha-snippets/resize-image.yaml\"}",
        (unsigned)LCD_IMAGE_MAX_BYTES, (unsigned long)req->content_len);
    size_t len = n > 0 ? (size_t)n : 0u;
    if (len >= sizeof(body)) {
        len = sizeof(body) - 1u;
    }
    httpd_resp_set_status(req, "413 Payload Too Large");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Connection", "close");
    if (fd >= 0) {
        int nodelay = 1;
        (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    }
    httpd_resp_send(req, body, len);
    log_request(req, 413, len);
    if (fd >= 0) {
        (void)httpd_sess_trigger_close(req->handle, fd);
    }
    return ESP_FAIL;
}

static esp_err_t local_api_dispatch(httpd_req_t *req)
{
    char path[64];
    char query[256];
    if (!split_uri(req, path, sizeof(path), query, sizeof(query))) {
        httpd_resp_send_err(req, HTTPD_414_URI_TOO_LONG, "invalid uri");
        log_request(req, 414, 0u);
        return ESP_OK;
    }

    const char *method = method_name(req->method);
    const nyx_route_t *route = find_local_api_route(path, method);
    if (!route || !route->fn) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "no route");
        log_request(req, 404, 0u);
        return ESP_OK;
    }

    // 64 KiB cap workaround for upstream ESP-Hosted SDIO bug:
    // https://github.com/espressif/esp-hosted-mcu/issues/144
    // (assert failed: sdio_rx_get_buffer sdio_drv.c:830).
    // Bigger receives stall and reset the board. HA-side resize keeps
    // us under the cap; remove this once upstream lands a fix.
    if (strcmp(path, "/lcd/image") == 0 && req->content_len > LCD_IMAGE_MAX_BYTES) {
        return reject_lcd_image_too_large(req);
    }

    uint8_t *body = NULL;
    size_t body_len = 0u;
    if (strcmp(path, "/lcd/image") == 0 || strcmp(path, "/lcd/clear") == 0) {
        ESP_LOGI(TAG, "dispatch begin path=%s method=%s content_len=%ld",
                 path, method, (long)req->content_len);
    }
    esp_err_t read_err = read_request_body(req, &body, &body_len);
    if (read_err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, esp_err_to_name(read_err));
        log_request(req, 500, 0u);
        return ESP_OK;
    }
    if (strcmp(path, "/lcd/image") == 0 || strcmp(path, "/lcd/clear") == 0) {
        ESP_LOGI(TAG, "dispatch body read path=%s body_len=%u",
                 path, (unsigned)body_len);
    }

    char content_type[64] = {0};
    size_t content_type_len = httpd_req_get_hdr_value_len(req, "Content-Type");
    if (content_type_len > 0u) {
        if (content_type_len >= sizeof(content_type)) {
            free(body);
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "content type too long");
            log_request(req, 400, body_len);
            return ESP_OK;
        }
        (void)httpd_req_get_hdr_value_str(req, "Content-Type", content_type,
                                          sizeof(content_type));
    }

    nyx_header_t headers[1] = {0};
    size_t header_count = 0u;
    if (content_type[0] != '\0') {
        headers[0].name = "content-type";
        headers[0].value = content_type;
        header_count = 1u;
    }
    nyx_proxy_request_t api_req = {
        .method = (char *)method,
        .path = path,
        .query = query[0] ? query : NULL,
        .body = body,
        .body_len = body_len,
        .header_count = header_count,
    };
    if (header_count > 0u) {
        api_req.headers[0] = headers[0];
    }

    size_t resp_cap = local_api_response_cap(path);
    char *resp = NULL;
    if (resp_cap <= LOCAL_API_SMALL_RESP_CAP) {
        resp = malloc(resp_cap);
    } else {
        resp = heap_caps_malloc(resp_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!resp) {
            resp_cap = LOCAL_API_FALLBACK_RESP_CAP;
            resp = malloc(resp_cap);
        }
    }
    if (!resp) {
        free(body);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "response alloc failed");
        log_request(req, 500, body_len);
        return ESP_OK;
    }

    char resp_type[64] = "application/json";
    size_t resp_len = 0u;
    if (strcmp(path, "/lcd/image") == 0 || strcmp(path, "/lcd/clear") == 0) {
        ESP_LOGI(TAG, "dispatch route call path=%s", path);
    }
    int status = local_api_route_with_logging(route, &api_req, resp, resp_cap,
                                              resp_type, sizeof(resp_type), &resp_len);
    free(body);
    if (strcmp(path, "/lcd/image") == 0 || strcmp(path, "/lcd/clear") == 0) {
        ESP_LOGI(TAG, "dispatch route done path=%s status=%d resp_len=%u",
                 path, status, (unsigned)resp_len);
    }

    char status_str[32];
    snprintf(status_str, sizeof(status_str), "%d %s", status,
             status == 200 ? "OK" :
             status == 202 ? "Accepted" :
             status == 400 ? "Bad Request" :
             status == 404 ? "Not Found" :
             status == 413 ? "Payload Too Large" :
             status == 415 ? "Unsupported Media Type" :
             status == 500 ? "Internal Server Error" : "Status");
    httpd_resp_set_status(req, status_str);
    httpd_resp_set_type(req, resp_type);
    esp_err_t send_err = httpd_resp_send(req, resp, resp_len);
    if (strcmp(path, "/lcd/image") == 0 || strcmp(path, "/lcd/clear") == 0) {
        ESP_LOGI(TAG, "dispatch response sent path=%s err=%s",
                 path, esp_err_to_name(send_err));
    }
    free(resp);
    log_request(req, status, resp_len);
    return send_err == ESP_OK ? ESP_OK : send_err;
}

static void ota_reboot(void *arg)
{
    vTaskDelay(pdMS_TO_TICKS(1500)); esp_restart();
}

static esp_err_t ota_image_post(httpd_req_t *req)
{
    char auth[192], expected[65];
    const char *key=nyx_creds_http_api_key();
    if (!key || !*key || httpd_req_get_hdr_value_str(req,"Authorization",auth,sizeof(auth)) != ESP_OK ||
        strncmp(auth,"Bearer ",7) || strlen(auth+7) != strlen(key)) {
        httpd_resp_set_status(req,"401 Unauthorized"); return httpd_resp_sendstr(req,"unauthorized");
    }
    unsigned diff=0;
    for (size_t i=0; i<strlen(key); ++i) diff |= (unsigned char)auth[7+i] ^ (unsigned char)key[i];
    if (diff) { httpd_resp_set_status(req,"401 Unauthorized"); return httpd_resp_sendstr(req,"unauthorized"); }
    const esp_partition_t *update=esp_ota_get_next_update_partition(NULL);
    if (!update || req->content_len <= 0 || (size_t)req->content_len > update->size ||
        httpd_req_get_hdr_value_str(req,"X-SHA256",expected,sizeof(expected)) != ESP_OK || strlen(expected)!=64) {
        httpd_resp_set_status(req,"400 Bad Request"); return httpd_resp_sendstr(req,"invalid size or X-SHA256");
    }
    for (int i=0; i<64; ++i) if (!isxdigit((unsigned char)expected[i])) {
        httpd_resp_set_status(req,"400 Bad Request"); return httpd_resp_sendstr(req,"invalid digest");
    }
    if (!panel_ota_try_lock()) { httpd_resp_set_status(req,"409 Conflict"); return httpd_resp_sendstr(req,"OTA busy"); }
    panel_ota_progress("receiving", ESP_OK);
    esp_ota_handle_t handle=0;
    esp_err_t err=esp_ota_begin(update,req->content_len,&handle);
    mbedtls_sha256_context sha;
    mbedtls_sha256_init(&sha); mbedtls_sha256_starts(&sha,0);
    char *buffer=malloc(4096);
    if (!buffer) err=ESP_ERR_NO_MEM;
    size_t received=0;
    int64_t deadline=esp_timer_get_time()+300000000LL;
    while (err==ESP_OK && received<(size_t)req->content_len) {
        size_t n=MIN(4096u,(size_t)req->content_len-received);
        int got=httpd_req_recv(req,buffer,n);
        if (got==HTTPD_SOCK_ERR_TIMEOUT && esp_timer_get_time()<deadline) continue;
        if (got<=0) { err=ESP_ERR_TIMEOUT; break; }
        mbedtls_sha256_update(&sha,(const uint8_t *)buffer,got);
        err=esp_ota_write(handle,buffer,got); received+=got;
    }
    free(buffer);
    uint8_t digest[32]; mbedtls_sha256_finish(&sha,digest); mbedtls_sha256_free(&sha);
    char actual[65];
    for (int i=0;i<32;++i) snprintf(actual+i*2,3,"%02x",digest[i]);
    if (err==ESP_OK) panel_ota_progress("verifying", ESP_OK);
    if (err==ESP_OK && strcasecmp(actual,expected)) err=ESP_ERR_INVALID_CRC;
    if (err==ESP_OK) { err=esp_ota_end(handle); handle=0; }
    if (err==ESP_OK) err=esp_ota_set_boot_partition(update);
    if (handle) esp_ota_abort(handle);
    httpd_resp_set_type(req,"application/json");
    if (err!=ESP_OK) {
        panel_ota_progress("failed", err);
        panel_ota_unlock();
        httpd_resp_set_status(req,err==ESP_ERR_INVALID_CRC?"422 Unprocessable Entity":"500 Internal Server Error");
        return httpd_resp_sendstr(req,"{\"ok\":false,\"error\":\"ota_image_rejected\"}");
    }
    panel_ota_progress("rebooting", ESP_OK);
    httpd_resp_set_status(req,"202 Accepted");
    esp_err_t sent=httpd_resp_sendstr(req,"{\"ok\":true,\"rebooting\":true}");
    xTaskCreate(ota_reboot,"ota_reboot",2048,NULL,5,NULL);
    return sent;
}

esp_err_t local_http_server_start(void)
{
    if (s_server) {
        return ESP_OK;
    }
    if (!s_snapshot_lock) {
        s_snapshot_lock = xSemaphoreCreateMutex();
        if (!s_snapshot_lock) {
            ESP_LOGE(TAG, "snapshot mutex alloc failed");
            return ESP_ERR_NO_MEM;
        }
    }
    if (!s_lcd_snapshot_lock) {
        s_lcd_snapshot_lock = xSemaphoreCreateMutex();
        if (!s_lcd_snapshot_lock) {
            ESP_LOGE(TAG, "LCD snapshot mutex alloc failed");
            return ESP_ERR_NO_MEM;
        }
    }

    cfg_lan_t cfg = {0};
    esp_err_t err = nvs_config_get_lan(&cfg);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "LAN HTTP config unavailable: %s", esp_err_to_name(err));
        nvs_config_defaults_lan(&cfg);
    }
    if (!cfg.enabled) {
        ESP_LOGI(TAG, "LAN HTTP server disabled by config");
        return ESP_OK;
    }

    httpd_config_t http_cfg = HTTPD_DEFAULT_CONFIG();
    http_cfg.server_port = cfg.port;
    http_cfg.ctrl_port = (uint16_t)(cfg.port == 32768u ? 32769u : 32768u);
    http_cfg.max_uri_handlers = 18;
    http_cfg.lru_purge_enable = true;
    http_cfg.stack_size = 16 * 1024;

    err = httpd_start(&s_server, &http_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start LAN HTTP server on port %u: %s",
                 (unsigned)cfg.port, esp_err_to_name(err));
        s_server = NULL;
        return err;
    }

    const httpd_uri_t ota_uri = { .uri = "/system/ota/image", .method = HTTP_POST, .handler = ota_image_post };
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &ota_uri));

    const httpd_uri_t root_uri = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = root_get,
        .user_ctx = NULL,
    };
    const httpd_uri_t status_uri = {
        .uri = "/status",
        .method = HTTP_GET,
        .handler = status_get,
        .user_ctx = NULL,
    };
    const httpd_uri_t snapshot_uri = {
        .uri = "/snapshot.jpg",
        .method = HTTP_GET,
        .handler = snapshot_get,
        .user_ctx = NULL,
    };
    const httpd_uri_t lcd_snapshot_uri = {
        .uri = "/lcd/snapshot.jpg",
        .method = HTTP_GET,
        .handler = lcd_snapshot_get,
        .user_ctx = NULL,
    };
    const httpd_uri_t lcd_screen_get_uri = {
        .uri = "/lcd/screen",
        .method = HTTP_GET,
        .handler = lcd_screen_get,
        .user_ctx = NULL,
    };
    const httpd_uri_t stream_status_get_uri = {
        .uri = "/stream/status",
        .method = HTTP_GET,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t stream_start_post_uri = {
        .uri = "/stream/start",
        .method = HTTP_POST,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t stream_stop_post_uri = {
        .uri = "/stream/stop",
        .method = HTTP_POST,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t stream_config_get_uri = {
        .uri = "/config/stream",
        .method = HTTP_GET,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t stream_config_post_uri = {
        .uri = "/config/stream",
        .method = HTTP_POST,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t mqtt_config_get_uri = {
        .uri = "/config/mqtt",
        .method = HTTP_GET,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t mqtt_config_post_uri = {
        .uri = "/config/mqtt",
        .method = HTTP_POST,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    // Register direct LAN LCD push routes; dispatch still validates against
    // local_api_lan_routes() before reading the request body.
    const httpd_uri_t lcd_image_post_uri = {
        .uri = "/lcd/image",
        .method = HTTP_POST,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };
    const httpd_uri_t lcd_clear_post_uri = {
        .uri = "/lcd/clear",
        .method = HTTP_POST,
        .handler = local_api_dispatch,
        .user_ctx = NULL,
    };

    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &root_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &status_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &snapshot_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &lcd_snapshot_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &lcd_screen_get_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &stream_status_get_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &stream_start_post_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &stream_stop_post_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &stream_config_get_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &stream_config_post_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &mqtt_config_get_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &mqtt_config_post_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &lcd_image_post_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &lcd_clear_post_uri));
    ESP_LOGI(TAG, "LAN HTTP server listening on port %u", (unsigned)cfg.port);
    return ESP_OK;
}
