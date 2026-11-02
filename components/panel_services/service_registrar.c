#include "service_registrar.h"

#include "cjson_tiny.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "nyx_creds.h"
#include "settings.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "haCam.reg";

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} response_buf_t;

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    response_buf_t *resp = (response_buf_t *)evt->user_data;
    if (evt->event_id != HTTP_EVENT_ON_DATA || !resp || !evt->data || evt->data_len <= 0) {
        return ESP_OK;
    }

    size_t available = (resp->cap > resp->len) ? (resp->cap - resp->len - 1u) : 0u;
    size_t copy_len = (size_t)evt->data_len;
    if (copy_len > available) {
        copy_len = available;
    }
    if (copy_len > 0u) {
        memcpy(resp->buf + resp->len, evt->data, copy_len);
        resp->len += copy_len;
        resp->buf[resp->len] = '\0';
    }
    return ESP_OK;
}

static int parse_slug_from_response(const char *json, char *out, size_t cap)
{
    cJSON *root = cJSON_Parse(json);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return -1;
    }

    cJSON *slug = cJSON_GetObjectItemCaseSensitive(root, "slug");
    int rc = -1;
    if (cJSON_IsString(slug) && slug->valuestring) {
        snprintf(out, cap, "%s", slug->valuestring);
        rc = 0;
    }
    cJSON_Delete(root);
    return rc;
}

static int parse_existing_slug(const char *json, const char *label, const char *node_id,
                               char *out, size_t cap)
{
    cJSON *root = cJSON_Parse(json);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return -1;
    }

    cJSON *keys = cJSON_GetObjectItemCaseSensitive(root, "keys");
    int rc = -1;
    if (cJSON_IsArray(keys)) {
        cJSON *item = NULL;
        cJSON_ArrayForEach(item, keys) {
            cJSON *item_label = cJSON_GetObjectItemCaseSensitive(item, "label");
            cJSON *item_node_id = cJSON_GetObjectItemCaseSensitive(item, "node_id");
            cJSON *item_slug = cJSON_GetObjectItemCaseSensitive(item, "slug");
            if (cJSON_IsString(item_label) && item_label->valuestring &&
                cJSON_IsString(item_node_id) && item_node_id->valuestring &&
                cJSON_IsString(item_slug) && item_slug->valuestring &&
                strcmp(item_label->valuestring, label) == 0 &&
                strcmp(item_node_id->valuestring, node_id) == 0) {
                snprintf(out, cap, "%s", item_slug->valuestring);
                rc = 0;
                break;
            }
        }
    }

    cJSON_Delete(root);
    return rc;
}

static char *build_body(const char *label)
{
    cJSON *body = cJSON_CreateObject();
    if (!body) {
        return NULL;
    }
    cJSON_AddStringToObject(body, "label", label);
    cJSON_AddStringToObject(body, "key_type", "http");
    cJSON_AddStringToObject(body, "endpoint_url", "http://nyx-local");
    cJSON_AddStringToObject(body, "auth_method", "none");
    cJSON_AddStringToObject(body, "auth_key_name", "Authorization");
    cJSON_AddStringToObject(body, "node_id", nyx_creds_node_id());
    char *printed = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    return printed;
}

static int run_request(esp_http_client_method_t method, const char *url, const char *body,
                       char *resp_storage, size_t resp_cap, int *status_out)
{
    if (!url || !resp_storage || resp_cap == 0u) {
        return -1;
    }

    resp_storage[0] = '\0';

    char auth[192];
    snprintf(auth, sizeof(auth), "Bearer %s", nyx_creds_http_api_key());

    esp_http_client_config_t cfg = {
        .url = url,
        .method = method,
        .timeout_ms = 10000,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };
    response_buf_t resp = {
        .buf = resp_storage,
        .len = 0,
        .cap = resp_cap,
    };
    cfg.event_handler = http_event_handler;
    cfg.user_data = &resp;

    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (!client) {
        return -1;
    }

    esp_http_client_set_header(client, "Authorization", auth);
    if (body) {
        esp_http_client_set_header(client, "Content-Type", "application/json");
        esp_http_client_set_post_field(client, body, strlen(body));
    }

    int rc = -1;
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        if (status_out) {
            *status_out = esp_http_client_get_status_code(client);
        }
        rc = 0;
    } else {
        ESP_LOGE(TAG, "service registration HTTP failed url=%s err=%s", url, esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
    return rc;
}

static int reuse_existing_slug(const char *label, char *resp_storage, size_t resp_cap)
{
    char url[224];
    snprintf(url, sizeof(url), "%s/api/v1/keys?label=%s", HA_CAM_NYX_BASE_URL, label);

    ESP_LOGI(TAG, "lookup slug label=%s node=%s", label, nyx_creds_node_id());
    int status = 0;
    if (run_request(HTTP_METHOD_GET, url, NULL, resp_storage, resp_cap, &status) != 0) {
        return -1;
    }
    if (status != 200) {
        ESP_LOGW(TAG, "service lookup failed status=%d body=%.200s", status, resp_storage);
        return -1;
    }

    char slug[80];
    if (parse_existing_slug(resp_storage, label, nyx_creds_node_id(), slug, sizeof(slug)) != 0) {
        ESP_LOGI(TAG, "no existing service for label=%s node=%s", label, nyx_creds_node_id());
        return 1;
    }

    if (nyx_creds_save_slug(slug) != 0) {
        return -1;
    }
    ESP_LOGI(TAG, "reused registered slug=%s label=%s", slug, label);
    return 0;
}

int service_registrar_run(const char *label)
{
    if (nyx_creds_service_slug()[0] != '\0') {
        ESP_LOGI(TAG, "slug already known: %s", nyx_creds_service_slug());
        return 0;
    }
    if (!label || label[0] == '\0') {
        return -1;
    }

    char *resp_storage = calloc(1, 4096);
    if (!resp_storage) {
        return -1;
    }

    int rc = reuse_existing_slug(label, resp_storage, 4096);
    if (rc == 0) {
        ESP_LOGI(TAG, "registrar decision=reuse slug=%s", nyx_creds_service_slug());
        free(resp_storage);
        return 0;
    }
    ESP_LOGI(TAG, "registrar decision=create label=%s", label);

    char url[192];
    snprintf(url, sizeof(url), "%s/api/v1/keys", HA_CAM_NYX_BASE_URL);

    char *body = build_body(label);
    if (!body) {
        free(resp_storage);
        return -1;
    }

    int status = 0;
    rc = -1;
    if (run_request(HTTP_METHOD_POST, url, body, resp_storage, 4096, &status) == 0) {
        char slug[80];
        if ((status == 200 || status == 201) &&
            parse_slug_from_response(resp_storage, slug, sizeof(slug)) == 0) {
            if (nyx_creds_save_slug(slug) == 0) {
                ESP_LOGI(TAG, "registered slug=%s", slug);
                rc = 0;
            }
        } else {
            ESP_LOGE(TAG, "service registration failed status=%d body=%.200s", status, resp_storage);
        }
    }

    free(resp_storage);
    cJSON_free(body);
    return rc;
}
