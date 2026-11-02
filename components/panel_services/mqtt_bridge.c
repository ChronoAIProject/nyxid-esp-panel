#include "mqtt_bridge.h"

#include "cjson_tiny.h"
#include "esp_app_desc.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "heap_diag.h"
#include "lcd_renderer.h"
#include "mqtt_client.h"
#include "net_manager.h"
#include "nvs_config.h"
#include "nyx_creds.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "mqtt_bridge";

#define MQTT_LCD_IMAGE_MAX_BYTES   (256u * 1024u)
#define MQTT_CLIENT_BUFFER_BYTES   (4u * 1024u)
#define MQTT_TEXT_MAX_BYTES        512u
#define MQTT_IMAGE_QUEUE_DEPTH     1u

typedef struct {
    uint8_t *jpeg;
    size_t len;
} image_job_t;

static struct {
    SemaphoreHandle_t lock;
    QueueHandle_t image_queue;
    TaskHandle_t image_task;
    esp_mqtt_client_handle_t client;
    cfg_mqtt_t cfg;
    bool started;
    bool connected;
    bool stopping;
    bool syslog_hooked;
    char client_id[96];
    char availability_topic[96];
    char syslog_topic[96];
    char image_set_topic[96];
    char text_set_topic[96];
    char clear_set_topic[96];
    char discovery_image_topic[128];
    char discovery_notify_topic[128];
    uint8_t *rx_image;
    size_t rx_image_len;
    size_t rx_image_total;
    char last_error[96];
} g;

static void syslog_notify(void)
{
    (void)0;
}

static void ensure_syslog_hook(void)
{
    if (g.syslog_hooked) {
        return;
    }
    ESP_LOGW(TAG, "MQTT syslog hook disabled; board stability takes precedence");
    g.syslog_hooked = true;
}

static void image_worker(void *arg)
{
    (void)arg;
    image_job_t job = {0};
    while (true) {
        if (xQueueReceive(g.image_queue, &job, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        int64_t start_us = esp_timer_get_time();
        esp_err_t err = lcd_renderer_push_image(job.jpeg, job.len, 0u);
        int64_t elapsed_ms = (esp_timer_get_time() - start_us) / 1000LL;
        ESP_LOGI(TAG, "lcd image push via mqtt len=%u result=%s elapsed_ms=%lld",
                 (unsigned)job.len, esp_err_to_name(err), (long long)elapsed_ms);
        heap_diag_log(TAG, "lcd_image_push_mqtt");
        heap_caps_free(job.jpeg);
        job.jpeg = NULL;
        job.len = 0u;
    }
}

static esp_err_t ensure_image_worker(void)
{
    if (!g.image_queue) {
        g.image_queue = xQueueCreate(MQTT_IMAGE_QUEUE_DEPTH, sizeof(image_job_t));
        if (!g.image_queue) {
            return ESP_ERR_NO_MEM;
        }
    }
    if (!g.image_task) {
        BaseType_t ok = xTaskCreate(image_worker, "mqtt_lcd_img", 8192, NULL, 4,
                                    &g.image_task);
        if (ok != pdPASS) {
            return ESP_ERR_NO_MEM;
        }
    }
    return ESP_OK;
}

static esp_err_t queue_image_job(uint8_t *jpeg, size_t len)
{
    if (!jpeg || len == 0u) {
        return ESP_ERR_INVALID_ARG;
    }
    esp_err_t err = ensure_image_worker();
    if (err != ESP_OK) {
        return err;
    }
    image_job_t job = {
        .jpeg = jpeg,
        .len = len,
    };
    if (xQueueSend(g.image_queue, &job, 0) == pdTRUE) {
        return ESP_OK;
    }
    image_job_t stale = {0};
    if (xQueueReceive(g.image_queue, &stale, 0) == pdTRUE) {
        heap_caps_free(stale.jpeg);
    }
    return xQueueSend(g.image_queue, &job, 0) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

static const char *device_slug(void)
{
    const char *slug = nyx_creds_service_slug();
    return (slug && slug[0] != '\0') ? slug : "esp32-cam-d2d72c-4";
}

static bool cfg_equal(const cfg_mqtt_t *a, const cfg_mqtt_t *b)
{
    return a && b &&
           a->enabled == b->enabled &&
           a->port == b->port &&
           strcmp(a->host, b->host) == 0 &&
           strcmp(a->user, b->user) == 0 &&
           strcmp(a->pass, b->pass) == 0 &&
           strcmp(a->base_topic, b->base_topic) == 0;
}

static void build_topics(const cfg_mqtt_t *cfg)
{
    const char *slug = device_slug();
    snprintf(g.client_id, sizeof(g.client_id), "haCam_%s", slug);
    snprintf(g.availability_topic, sizeof(g.availability_topic), "%s/availability", cfg->base_topic);
    snprintf(g.syslog_topic, sizeof(g.syslog_topic), "%s/syslog", cfg->base_topic);
    snprintf(g.image_set_topic, sizeof(g.image_set_topic), "%s/lcd/image/set", cfg->base_topic);
    snprintf(g.text_set_topic, sizeof(g.text_set_topic), "%s/lcd/text/set", cfg->base_topic);
    snprintf(g.clear_set_topic, sizeof(g.clear_set_topic), "%s/lcd/clear/set", cfg->base_topic);
    snprintf(g.discovery_image_topic, sizeof(g.discovery_image_topic),
             "homeassistant/image/haCam_%s/config", slug);
    snprintf(g.discovery_notify_topic, sizeof(g.discovery_notify_topic),
             "homeassistant/notify/haCam_%s/config", slug);
}

static void add_device_block(cJSON *root)
{
    cJSON *dev = cJSON_AddObjectToObject(root, "device");
    if (!dev) {
        return;
    }
    cJSON *ids = cJSON_AddArrayToObject(dev, "identifiers");
    if (ids) {
        char identifier[96];
        snprintf(identifier, sizeof(identifier), "haCam_%s", device_slug());
        cJSON_AddItemToArray(ids, cJSON_CreateString(identifier));
    }
    cJSON_AddStringToObject(dev, "sw_version", esp_app_get_description()->version);
    cJSON_AddStringToObject(dev, "name", g.client_id);
    cJSON_AddStringToObject(dev, "manufacturer", "ChronoAI");
    cJSON_AddStringToObject(dev, "model", "voice_presence_ha_camera");

    net_manager_snapshot_t net = net_manager_snapshot();
    const char *ip = net.eth_ip ? net.eth_ip_addr : (net.wifi_ip ? net.wifi_ip_addr : NULL);
    if (ip && ip[0] != '\0') {
        char url[32];
        snprintf(url, sizeof(url), "http://%s/", ip);
        cJSON_AddStringToObject(dev, "configuration_url", url);
    }
}

static void normalized_slug(char *out, size_t cap)
{
    const char *slug = device_slug();
    size_t j = 0u;
    if (!out || cap == 0u) {
        return;
    }
    for (size_t i = 0u; slug[i] != '\0' && j + 1u < cap; ++i) {
        char c = slug[i];
        out[j++] = (c == '-') ? '_' : c;
    }
    out[j] = '\0';
}

static char *discovery_notify_json(void)
{
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        return NULL;
    }
    char name[128];
    char unique_id[128];
    char default_entity_id[128];
    char norm[80];
    normalized_slug(norm, sizeof(norm));
    snprintf(name, sizeof(name), "haCam %s LCD", device_slug());
    snprintf(unique_id, sizeof(unique_id), "haCam_%s_lcd", norm);
    snprintf(default_entity_id, sizeof(default_entity_id), "notify.hacam_%s_lcd", norm);
    cJSON_AddStringToObject(root, "name", name);
    cJSON_AddStringToObject(root, "unique_id", unique_id);
    cJSON_AddStringToObject(root, "default_entity_id", default_entity_id);
    cJSON_AddStringToObject(root, "command_topic", g.text_set_topic);
    // No command_template: HA's notify.send_message publishes the literal
    // message text as the topic payload by default. With "{{ message }}"
    // some HA versions render against an empty context and publish (null),
    // which our /lcd/text/set handler then drops silently.
    cJSON_AddStringToObject(root, "availability_topic", g.availability_topic);
    add_device_block(root);
    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return json;
}

static void publish_discovery(void)
{
    char *notify = discovery_notify_json();
    if (!notify) {
        ESP_LOGE(TAG, "discovery json oom");
        return;
    }
    (void)esp_mqtt_client_publish(g.client, g.discovery_image_topic, "", 0, 1, 1);
    (void)esp_mqtt_client_publish(g.client, g.discovery_notify_topic, notify, 0, 1, 1);
    ESP_LOGI(TAG, "discovery published");
    free(notify);
}

static void subscribe_commands(void)
{
    (void)esp_mqtt_client_subscribe(g.client, g.image_set_topic, 1);
    ESP_LOGI(TAG, "subscribed to lcd/image/set");
    (void)esp_mqtt_client_subscribe(g.client, g.text_set_topic, 1);
    ESP_LOGI(TAG, "subscribed to lcd/text/set");
    (void)esp_mqtt_client_subscribe(g.client, g.clear_set_topic, 1);
    ESP_LOGI(TAG, "subscribed to lcd/clear/set");
}

static bool topic_equals(const esp_mqtt_event_t *event, const char *topic)
{
    return event && topic &&
           event->topic_len == (int)strlen(topic) &&
           strncmp(event->topic, topic, (size_t)event->topic_len) == 0;
}

static void reset_rx_image(void)
{
    heap_caps_free(g.rx_image);
    g.rx_image = NULL;
    g.rx_image_len = 0u;
    g.rx_image_total = 0u;
}

static void handle_image_data(const esp_mqtt_event_t *event)
{
    if (!event || event->total_data_len <= 0 || event->data_len < 0 ||
        event->current_data_offset < 0) {
        return;
    }
    size_t total = (size_t)event->total_data_len;
    size_t offset = (size_t)event->current_data_offset;
    size_t len = (size_t)event->data_len;
    if (offset == 0u || total > MQTT_CLIENT_BUFFER_BYTES) {
        ESP_LOGI(TAG, "lcd image rx chunk offset=%u len=%u total=%u",
                 (unsigned)offset, (unsigned)len, (unsigned)total);
    }
    if (total > MQTT_LCD_IMAGE_MAX_BYTES || offset + len > total) {
        ESP_LOGW(TAG, "dropping image payload total=%u", (unsigned)total);
        reset_rx_image();
        return;
    }
    if (offset == 0u) {
        reset_rx_image();
        g.rx_image = heap_caps_malloc(total, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!g.rx_image) {
            g.rx_image = heap_caps_malloc(total, MALLOC_CAP_8BIT);
        }
        g.rx_image_total = total;
    }
    if (!g.rx_image || g.rx_image_total != total) {
        return;
    }
    memcpy(g.rx_image + offset, event->data, len);
    if (offset + len > g.rx_image_len) {
        g.rx_image_len = offset + len;
    }
    if (g.rx_image_len == total) {
        uint8_t *image = g.rx_image;
        size_t image_len = g.rx_image_len;
        g.rx_image = NULL;
        g.rx_image_len = 0u;
        g.rx_image_total = 0u;
        esp_err_t err = queue_image_job(image, image_len);
        if (err != ESP_OK) {
            heap_caps_free(image);
        }
        ESP_LOGI(TAG, "lcd image queued via mqtt len=%u result=%s",
                 (unsigned)image_len, esp_err_to_name(err));
    }
}

static void handle_text_data(const esp_mqtt_event_t *event)
{
    if (!event || event->total_data_len <= 0) {
        return;
    }
    size_t len = (size_t)event->total_data_len;
    if (len > MQTT_TEXT_MAX_BYTES) {
        len = MQTT_TEXT_MAX_BYTES;
    }
    char *text = malloc(len + 1u);
    if (!text) {
        return;
    }
    size_t copy = (size_t)event->data_len;
    if (copy > len) {
        copy = len;
    }
    memcpy(text, event->data, copy);
    text[copy] = '\0';
    esp_err_t err = lcd_renderer_push_text(text, "medium", "#00ff41", "#000000", "center", 0u);
    ESP_LOGI(TAG, "lcd text push via mqtt len=%u result=%s", (unsigned)copy, esp_err_to_name(err));
    free(text);
}

static void handle_data(const esp_mqtt_event_t *event)
{
    if (event && event->current_data_offset > 0 && g.rx_image) {
        handle_image_data(event);
    } else if (topic_equals(event, g.image_set_topic)) {
        handle_image_data(event);
    } else if (topic_equals(event, g.text_set_topic)) {
        if (event->current_data_offset == 0) {
            handle_text_data(event);
        }
    } else if (topic_equals(event, g.clear_set_topic)) {
        if (event->current_data_offset == 0) {
            lcd_renderer_clear();
            ESP_LOGI(TAG, "lcd clear via mqtt");
        }
    }
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data)
{
    (void)handler_args;
    (void)base;
    esp_mqtt_event_t *event = (esp_mqtt_event_t *)event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        g.connected = true;
        g.last_error[0] = '\0';
        ESP_LOGI(TAG, "connected");
        (void)esp_mqtt_client_publish(g.client, g.availability_topic, "online", 0, 1, 1);
        publish_discovery();
        subscribe_commands();
        syslog_notify();
        heap_diag_log(TAG, "mqtt_connected");
        break;
    case MQTT_EVENT_DISCONNECTED:
        g.connected = false;
        ESP_LOGW(TAG, "disconnected");
        reset_rx_image();
        heap_diag_log(TAG, "mqtt_disconnected");
        break;
    case MQTT_EVENT_ERROR:
        if (event && event->error_handle) {
            snprintf(g.last_error, sizeof(g.last_error), "type=%d connect_rc=%d esp_err=0x%x sock_errno=%d",
                     event->error_handle->error_type,
                     event->error_handle->connect_return_code,
                     (unsigned)event->error_handle->esp_tls_last_esp_err,
                     event->error_handle->esp_transport_sock_errno);
            ESP_LOGW(TAG, "error type=%d connect_rc=%d esp_err=0x%x sock_errno=%d",
                     event->error_handle->error_type,
                     event->error_handle->connect_return_code,
                     (unsigned)event->error_handle->esp_tls_last_esp_err,
                     event->error_handle->esp_transport_sock_errno);
        } else {
            snprintf(g.last_error, sizeof(g.last_error), "mqtt_error");
            ESP_LOGW(TAG, "error");
        }
        break;
    case MQTT_EVENT_DATA:
        handle_data(event);
        break;
    default:
        break;
    }
}

static esp_err_t start_locked(const cfg_mqtt_t *cfg)
{
    if (!cfg->enabled) {
        ESP_LOGI(TAG, "disabled, skipping start");
        g.cfg = *cfg;
        g.started = false;
        g.stopping = false;
        snprintf(g.last_error, sizeof(g.last_error), "disabled");
        return ESP_OK;
    }
    esp_err_t worker_err = ensure_image_worker();
    if (worker_err != ESP_OK) {
        ESP_LOGE(TAG, "image worker start failed: %s", esp_err_to_name(worker_err));
        return worker_err;
    }
    build_topics(cfg);
    ensure_syslog_hook();
    g.stopping = false;
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.hostname = cfg->host,
        .broker.address.port = cfg->port,
        .broker.address.transport = MQTT_TRANSPORT_OVER_TCP,
        .credentials.username = cfg->user[0] ? cfg->user : NULL,
        .credentials.client_id = g.client_id,
        .credentials.authentication.password = cfg->pass[0] ? cfg->pass : NULL,
        .session.last_will.topic = g.availability_topic,
        .session.last_will.msg = "offline",
        .session.last_will.qos = 1,
        .session.last_will.retain = 1,
        .session.keepalive = 30,
        .network.reconnect_timeout_ms = 1000,
        .network.timeout_ms = 10000,
        .task.stack_size = 8192,
        .buffer.size = MQTT_CLIENT_BUFFER_BYTES,
        .buffer.out_size = MQTT_CLIENT_BUFFER_BYTES,
    };
    g.client = esp_mqtt_client_init(&mqtt_cfg);
    if (!g.client) {
        ESP_LOGE(TAG, "client init failed");
        snprintf(g.last_error, sizeof(g.last_error), "client_init_failed");
        return ESP_FAIL;
    }
    esp_err_t err = esp_mqtt_client_register_event(g.client, MQTT_EVENT_ANY, mqtt_event_handler, NULL);
    if (err == ESP_OK) {
        err = esp_mqtt_client_start(g.client);
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "start failed: %s", esp_err_to_name(err));
        snprintf(g.last_error, sizeof(g.last_error), "start:%s", esp_err_to_name(err));
        esp_mqtt_client_destroy(g.client);
        g.client = NULL;
        return err;
    }
    g.cfg = *cfg;
    g.started = true;
    g.last_error[0] = '\0';
    ESP_LOGI(TAG, "start host=%s port=%u base_topic=%s", cfg->host, (unsigned)cfg->port, cfg->base_topic);
    return ESP_OK;
}

esp_err_t mqtt_bridge_start(void)
{
    if (!g.lock) {
        g.lock = xSemaphoreCreateMutex();
        if (!g.lock) {
            return ESP_ERR_NO_MEM;
        }
        cfg_mqtt_t current = {0};
        if (nvs_config_get_mqtt(&current) == ESP_OK) {
            g.cfg = current;
        }
    }
    cfg_mqtt_t cfg = {0};
    esp_err_t err = nvs_config_get_mqtt(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "config read failed: %s", esp_err_to_name(err));
        return err;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    if (g.client || g.started) {
        xSemaphoreGive(g.lock);
        return ESP_OK;
    }
    err = start_locked(&cfg);
    xSemaphoreGive(g.lock);
    return err;
}

void mqtt_bridge_stop(void)
{
    if (!g.lock) {
        return;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    g.stopping = true;
    g.started = false;
    g.connected = false;
    reset_rx_image();
    esp_mqtt_client_handle_t client = g.client;
    g.client = NULL;
    xSemaphoreGive(g.lock);

    if (client) {
        (void)esp_mqtt_client_publish(client, g.availability_topic, "offline", 0, 1, 1);
        (void)esp_mqtt_client_stop(client);
        (void)esp_mqtt_client_destroy(client);
    }
}

void mqtt_bridge_reload(void)
{
    cfg_mqtt_t cfg = {0};
    if (nvs_config_get_mqtt(&cfg) != ESP_OK) {
        return;
    }
    if (!g.lock) {
        (void)mqtt_bridge_start();
        return;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    bool same = cfg_equal(&cfg, &g.cfg);
    xSemaphoreGive(g.lock);
    if (same) {
        ESP_LOGI(TAG, "reload skipped; config unchanged");
        return;
    }
    ESP_LOGI(TAG, "reload applying config");
    mqtt_bridge_stop();
    (void)mqtt_bridge_start();
}

bool mqtt_bridge_is_started(void)
{
    if (!g.lock) {
        return false;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    bool started = g.started;
    xSemaphoreGive(g.lock);
    return started;
}

bool mqtt_bridge_is_connected(void)
{
    if (!g.lock) {
        return false;
    }
    xSemaphoreTake(g.lock, portMAX_DELAY);
    bool connected = g.connected;
    xSemaphoreGive(g.lock);
    return connected;
}

void mqtt_bridge_get_status(mqtt_bridge_status_t *status)
{
    if (!status) {
        return;
    }
    memset(status, 0, sizeof(*status));

    cfg_mqtt_t cfg = {0};
    if (nvs_config_get_mqtt(&cfg) != ESP_OK) {
        nvs_config_defaults_mqtt(&cfg);
    }
    status->enabled = cfg.enabled;
    snprintf(status->host, sizeof(status->host), "%s", cfg.host);

    if (!g.lock) {
        snprintf(status->state, sizeof(status->state), "disconnected");
        snprintf(status->last_error, sizeof(status->last_error), "%s",
                 cfg.enabled ? "not_started" : "disabled");
        return;
    }

    xSemaphoreTake(g.lock, portMAX_DELAY);
    status->started = g.started;
    status->connected = g.connected;
    snprintf(status->state, sizeof(status->state), "%s",
             g.connected ? "connected" : "disconnected");
    snprintf(status->last_error, sizeof(status->last_error), "%s",
             g.last_error[0] ? g.last_error : "");
    xSemaphoreGive(g.lock);

    if (!cfg.enabled) {
        snprintf(status->last_error, sizeof(status->last_error), "disabled");
    } else if (!status->started && status->last_error[0] == '\0') {
        snprintf(status->last_error, sizeof(status->last_error), "not_started");
    }
}
