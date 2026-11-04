/* NyxID account device login, used only to provision a WSS control node.
 * User-session tokens stay in RAM. A hardware client API key is never a node token.
 */
#include "device_code.h"
#include "device_login_proto.h"
#include "devices.h"
#include "net_manager.h"
#include "nyx_creds.h"
#include "nyx_proto.h"
#include "panel_voice.h"
#include "panel_access.h"
#include "settings.h"
#include "cJSON.h"
#include "esp_app_desc.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_timer.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#define RESPONSE_CAP 65536
static portMUX_TYPE state_lock = portMUX_INITIALIZER_UNLOCKED;
static device_login_snapshot_t state;
static int64_t deadline_us;
static atomic_bool cancelled;

static void erase(void *data, size_t size)
{
    volatile unsigned char *p = data;
    while (size--) *p++ = 0;
}

static void set_state(device_login_state_t next, const char *message)
{
    portENTER_CRITICAL(&state_lock);
    state.state = next;
    snprintf(state.message, sizeof(state.message), "%s", message);
    if (next != DEVICE_LOGIN_WAITING) {
        state.user_code[0] = 0;
        state.verification_uri[0] = 0;
        state.verification_uri_complete[0] = 0;
        deadline_us = 0;
    }
    portEXIT_CRITICAL(&state_lock);
}

void device_code_snapshot(device_login_snapshot_t *out)
{
    if (!out) return;
    portENTER_CRITICAL(&state_lock);
    *out = state;
    int64_t left = deadline_us ? deadline_us - esp_timer_get_time() : 0;
    out->seconds_remaining = left > 0 ? (uint32_t)((left + 999999) / 1000000) : 0;
    portEXIT_CRITICAL(&state_lock);
}

void device_code_cancel(void) { atomic_store(&cancelled, true); }

static bool wait_seconds(uint32_t seconds)
{
    for (uint32_t i = 0; i < seconds * 10; ++i) {
        if (atomic_load(&cancelled)) return false;
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    return !atomic_load(&cancelled);
}

typedef struct { char *data; size_t len; bool overflow; } response_t;
static esp_err_t http_event(esp_http_client_event_t *event)
{
    response_t *response = event->user_data;
    if (event->event_id == HTTP_EVENT_ON_DATA && event->data_len > 0) {
        size_t len = (size_t)event->data_len;
        if (len >= RESPONSE_CAP - response->len) response->overflow = true;
        else if (!response->overflow) {
            memcpy(response->data + response->len, event->data, len);
            response->len += len;
            response->data[response->len] = 0;
        }
    }
    return ESP_OK;
}

static int request(esp_http_client_method_t method, const char *path,
                   const char *access_token, const cJSON *body, char *response)
{
    erase(response, RESPONSE_CAP);
    char url[384];
    const char *base = !strncmp(path, "/auth/agent-key/", 16) ? NYX_AGENT_BASE_URL : HA_CAM_NYX_BASE_URL;
    if (snprintf(url, sizeof(url), "%s/api/v1%s", base, path) >= sizeof(url)) return 0;
    response_t data = {.data = response};
    esp_http_client_config_t config = {
        .url = url, .method = method, .timeout_ms = 12000,
        // An account JWT can exceed ESP-IDF's default 512-byte header buffer.
        // Each header must fit whole; otherwise Authorization is never sent.
        .buffer_size_tx = access_token ? 10240 : 1024,
        .disable_auto_redirect = true, .crt_bundle_attach = esp_crt_bundle_attach,
        .event_handler = http_event, .user_data = &data,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) return 0;
    char *authorization = NULL;
    if (access_token) {
        authorization = malloc(strlen(access_token) + 8);
        if (!authorization) { esp_http_client_cleanup(client); return 0; }
        sprintf(authorization, "Bearer %s", access_token);
        esp_http_client_set_header(client, "Authorization", authorization);
    }
    char *json = body ? cJSON_PrintUnformatted(body) : NULL;
    if (body && !json) {
        if (authorization) { erase(authorization, strlen(authorization)); free(authorization); }
        esp_http_client_cleanup(client);
        return 0;
    }
    esp_http_client_set_header(client, "Content-Type", "application/json");
    if (json) esp_http_client_set_post_field(client, json, strlen(json));
    esp_err_t error = esp_http_client_perform(client);
    int status = error == ESP_OK && !data.overflow ? esp_http_client_get_status_code(client) : 0;
    esp_http_client_cleanup(client);
    if (json) { erase(json, strlen(json)); free(json); }
    if (authorization) { erase(authorization, strlen(authorization)); free(authorization); }
    return status;
}

static bool json_copy(const cJSON *json, const char *key, char *out, size_t cap)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, key);
    if (!cJSON_IsString(item) || !item->valuestring[0] || strlen(item->valuestring) >= cap) return false;
    strcpy(out, item->valuestring);
    return true;
}

static bool valid_node(const char *id)
{
    if (strlen(id) != 36) return false;
    for (size_t i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) { if (id[i] != '-') return false; }
        else if (!isxdigit((unsigned char)id[i])) return false;
    }
    return true;
}

static bool valid_credentials(const nyx_binding_t *binding)
{
    if (!valid_node(binding->node_id) || strncmp(binding->auth_token, "nyx_nauth_", 10) ||
        strlen(binding->signing_secret) != 64) return false;
    for (size_t i = 0; i < 64; ++i) if (!isxdigit((unsigned char)binding->signing_secret[i])) return false;
    return true;
}

static bool save_stage(const nyx_binding_t *binding)
{
    nvs_handle_t h;
    if (nvs_open(NYX_NODE_SETUP_NVS_NAMESPACE, NVS_READWRITE, &h) != ESP_OK) return false;
    esp_err_t err = nvs_set_blob(h, "pending_v1", binding, sizeof(*binding));
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err == ESP_OK;
}

static void load_stage(nyx_binding_t *binding)
{
    memset(binding, 0, sizeof(*binding));
    nvs_handle_t h;
    if (nvs_open(NYX_NODE_SETUP_NVS_NAMESPACE, NVS_READONLY, &h) != ESP_OK) return;
    size_t size = sizeof(*binding);
    esp_err_t err = nvs_get_blob(h, "pending_v1", binding, &size);
    nvs_close(h);
    if (err != ESP_OK || size != sizeof(*binding) ||
        !memchr(binding->node_id, 0, sizeof(binding->node_id)) ||
        !memchr(binding->auth_token, 0, sizeof(binding->auth_token)) ||
        !memchr(binding->signing_secret, 0, sizeof(binding->signing_secret)) ||
        !memchr(binding->slug, 0, sizeof(binding->slug)) || !valid_node(binding->node_id))
        memset(binding, 0, sizeof(*binding));
}

typedef struct {
    EventGroupHandle_t events;
    char message[4096];
} registration_t;

static void websocket_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)base;
    registration_t *registration = arg;
    esp_websocket_event_data_t *event = data;
    if (id == WEBSOCKET_EVENT_CONNECTED) xEventGroupSetBits(registration->events, 1);
    else if (id == WEBSOCKET_EVENT_ERROR || id == WEBSOCKET_EVENT_DISCONNECTED)
        xEventGroupSetBits(registration->events, 4);
    else if (id == WEBSOCKET_EVENT_DATA && event && (event->op_code == 1 || event->op_code == 0)) {
        if (event->payload_len <= 0 || event->payload_len >= sizeof(registration->message) ||
            event->payload_offset < 0 || event->data_len < 0 ||
            event->payload_offset + event->data_len > event->payload_len) {
            xEventGroupSetBits(registration->events, 4);
            return;
        }
        memcpy(registration->message + event->payload_offset, event->data_ptr, event->data_len);
        if (event->payload_offset + event->data_len == event->payload_len) {
            registration->message[event->payload_len] = 0;
            xEventGroupSetBits(registration->events, 2);
        }
    }
}

static bool wait_event(registration_t *registration, EventBits_t bit)
{
    for (int i = 0; i < 20 && !atomic_load(&cancelled); ++i) {
        EventBits_t flags = xEventGroupWaitBits(registration->events, bit | 4, pdFALSE, pdFALSE, pdMS_TO_TICKS(1000));
        if (flags & bit) return true;
        if (flags & 4) return false;
    }
    return false;
}

static bool register_node(const char *registration_token, nyx_binding_t *binding)
{
    registration_t *registration = calloc(1, sizeof(*registration));
    if (!registration) return false;
    registration->events = xEventGroupCreate();
    if (!registration->events) { free(registration); return false; }
    esp_websocket_client_config_t config = {
        .uri = HA_CAM_NYX_WSS_URL, .disable_auto_reconnect = true,
        .network_timeout_ms = 10000, .buffer_size = 2048, .crt_bundle_attach = esp_crt_bundle_attach,
    };
    esp_websocket_client_handle_t ws = esp_websocket_client_init(&config);
    bool ok = false;
    if (ws) {
        esp_websocket_register_events(ws, WEBSOCKET_EVENT_ANY, websocket_event, registration);
        if (esp_websocket_client_start(ws) == ESP_OK && wait_event(registration, 1)) {
            char frame[512];
            size_t size = nyx_proto_build_register_token(frame, sizeof(frame), registration_token);
            if (size && !atomic_load(&cancelled) &&
                esp_websocket_client_send_text(ws, frame, size, pdMS_TO_TICKS(3000)) >= 0 && wait_event(registration, 2)) {
                cJSON *json = cJSON_Parse(registration->message);
                const cJSON *type = cJSON_GetObjectItemCaseSensitive(json, "type");
                char node[96];
                ok = cJSON_IsString(type) && !strcmp(type->valuestring, "register_ok") &&
                    json_copy(json, "node_id", node, sizeof(node)) && !strcmp(node, binding->node_id) &&
                    json_copy(json, "auth_token", binding->auth_token, sizeof(binding->auth_token)) &&
                    json_copy(json, "signing_secret", binding->signing_secret, sizeof(binding->signing_secret)) &&
                    valid_credentials(binding);
                cJSON_Delete(json);
            }
            erase(frame, sizeof(frame));
        }
        esp_websocket_client_stop(ws);
        esp_websocket_client_destroy(ws);
    }
    vEventGroupDelete(registration->events);
    erase(registration, sizeof(*registration));
    free(registration);
    return ok;
}

static bool setup_node(const char *access, char *response, nyx_binding_t *binding)
{
    load_stage(binding);
    char path[160];
    if (!binding->node_id[0] && nyx_creds_have_nvs_credentials()) {
        // Reauthorizing the same account reuses the existing board and service.
        snprintf(path, sizeof(path), "/nodes/%s", nyx_creds_node_id());
        if (request(HTTP_METHOD_GET, path, access, NULL, response) == 200) {
            snprintf(binding->node_id, sizeof(binding->node_id), "%s", nyx_creds_node_id());
            snprintf(binding->auth_token, sizeof(binding->auth_token), "%s", nyx_creds_api_key());
            snprintf(binding->signing_secret, sizeof(binding->signing_secret), "%s", nyx_creds_signing_secret());
            snprintf(binding->slug, sizeof(binding->slug), "%s", nyx_creds_service_slug());
            if (!valid_credentials(binding)) memset(binding, 0, sizeof(*binding));
        }
    }
    if (binding->node_id[0]) {
        snprintf(path, sizeof(path), "/nodes/%s", binding->node_id);
        int status = request(HTTP_METHOD_GET, path, access, NULL, response);
        if (status == 404) memset(binding, 0, sizeof(*binding));
        else if (status != 200) return false;
        else if (!valid_credentials(binding)) {
            // token_id survives a lost register_ok; a fresh human grant recovers it.
            snprintf(path, sizeof(path), "/nodes/%s/rotate-token", binding->node_id);
            if (request(HTTP_METHOD_POST, path, access, NULL, response) != 200) return false;
            cJSON *json = cJSON_Parse(response);
            bool ok = json_copy(json, "auth_token", binding->auth_token, sizeof(binding->auth_token)) &&
                json_copy(json, "signing_secret", binding->signing_secret, sizeof(binding->signing_secret)) &&
                valid_credentials(binding);
            cJSON_Delete(json);
            if (!ok || !save_stage(binding)) return false;
        }
    }
    if (binding->node_id[0]) return valid_credentials(binding);
    if (atomic_load(&cancelled)) return false;
    cJSON *body = cJSON_CreateObject();
    if (!body) return false;
    cJSON_AddStringToObject(body, "name", devices_self()->label);
    int status = request(HTTP_METHOD_POST, "/nodes/register-token", access, body, response);
    cJSON_Delete(body);
    if (status != 200) return false;
    char token[256] = "";
    cJSON *json = cJSON_Parse(response);
    bool ok = json_copy(json, "token_id", binding->node_id, sizeof(binding->node_id)) &&
        valid_node(binding->node_id) && json_copy(json, "token", token, sizeof(token)) && save_stage(binding);
    cJSON_Delete(json);
    if (ok) ok = register_node(token, binding) && save_stage(binding);
    erase(token, sizeof(token));
    return ok;
}

static bool service_matches(const char *response, const nyx_binding_t *binding, const char *expected_slug)
{
    cJSON *json = cJSON_Parse(response);
    char node[96], slug[80], method[16], status[24], endpoint[128];
    bool ok = json_copy(json, "node_id", node, sizeof(node)) && !strcmp(node, binding->node_id) &&
        json_copy(json, "slug", slug, sizeof(slug)) && !strcmp(slug,expected_slug) &&
        json_copy(json, "auth_method", method, sizeof(method)) && !strcmp(method, "none") &&
        json_copy(json, "status", status, sizeof(status)) && !strcmp(status, "active") &&
        json_copy(json, "endpoint_url", endpoint, sizeof(endpoint)) && !strcmp(endpoint, "http://127.0.0.1:80");
    cJSON_Delete(json);
    return ok;
}

static bool setup_one_service(const char *access, char *response, const nyx_binding_t *binding,
                              const char *slug, const char *label, bool daily)
{
    char path[192], spec_url[384];
    if (daily && snprintf(spec_url, sizeof(spec_url),
        "%s/api/v1/catalog-specs/nyx-panel/openapi.json", HA_CAM_NYX_BASE_URL) >= sizeof(spec_url)) return false;
    snprintf(path, sizeof(path), "/keys/%s", slug);
    int status = request(HTTP_METHOD_GET, path, access, NULL, response);
    if (status == 200) {
        if (!service_matches(response, binding, slug)) return false;
        if (!daily) return true;
        cJSON *stored = cJSON_Parse(response);
        const cJSON *spec = cJSON_GetObjectItemCaseSensitive(stored, "openapi_spec_url");
        bool has_spec = cJSON_IsString(spec) && spec->valuestring[0];
        cJSON_Delete(stored);
        // Preserve an owner-supplied schema. Resume older registrations that
        // predate typed tool discovery through the normal connection API.
        if (has_spec) return true;
        cJSON *update = cJSON_CreateObject();
        if (!update) return false;
        cJSON_AddStringToObject(update, "openapi_spec_url", spec_url);
        status = request(HTTP_METHOD_PUT, path, access, update, response);
        cJSON_Delete(update);
        return status == 200;
    }
    if (status != 404 || atomic_load(&cancelled)) return false;
    cJSON *body = cJSON_CreateObject();
    if (!body) return false;
    cJSON_AddStringToObject(body, "label", label);
    cJSON_AddStringToObject(body, "slug", slug);
    cJSON_AddStringToObject(body, "endpoint_url", "http://127.0.0.1:80");
    cJSON_AddStringToObject(body, "auth_method", "none");
    cJSON_AddStringToObject(body, "auth_key_name", "");
    cJSON_AddStringToObject(body, "node_id", binding->node_id);
    if (daily) cJSON_AddStringToObject(body, "openapi_spec_url", spec_url);
    status = request(HTTP_METHOD_POST, "/keys", access, body, response);
    cJSON_Delete(body);
    if (status != 200 && status != 201 && status != 409) return false;
    // Verify the stored routing and ownership, including after a slug conflict.
    return request(HTTP_METHOD_GET, path, access, NULL, response) == 200 && service_matches(response, binding, slug);
}
static bool setup_service(const char *access, char *response, nyx_binding_t *binding)
{
    if(!binding->slug[0])snprintf(binding->slug,sizeof(binding->slug),"%s-%.8s",devices_self()->label,binding->node_id);
    char management[80];
    if(!panel_management_slug(binding->slug,management,sizeof(management)))return false;
    return setup_one_service(access,response,binding,binding->slug,"Nyx Panel",true) &&
        !atomic_load(&cancelled) && setup_one_service(access,response,binding,management,"Nyx Panel Management",false);
}

static void login_task(void *arg)
{
    bool node_setup = (uintptr_t)arg != 0;
    char *response = calloc(1, RESPONSE_CAP);
    char *access = NULL;
    device_login_request_t login = {0};
    nyx_binding_t binding = {0};
    bool ready = false;
    const char *failure = "Could not start sign-in. Check the network and try again.";
    if (!response) goto done;
    cJSON *body = cJSON_CreateObject();
    if (!body) goto done;
    cJSON_AddStringToObject(body, "client_label", devices_self()->label);
    cJSON_AddStringToObject(body, "client_app", "nyxid-esp-panel");
    cJSON_AddStringToObject(body, "client_platform", "esp32p4");
    cJSON_AddStringToObject(body, "client_model", "Waveshare ESP32-P4 10.1");
    int status = request(HTTP_METHOD_POST, node_setup ? "/auth/device/request" : "/auth/agent-key/request", NULL, body, response);
    cJSON_Delete(body);
    if (status != 200 || !(node_setup ? device_login_parse_request_at(response, NYX_NODE_VERIFICATION_URI, &login) :
        device_agent_parse_request_at(response, NYX_AGENT_VERIFICATION_URI, &login))) goto done;
    if (atomic_load(&cancelled)) goto done;
    int64_t expires_at = esp_timer_get_time() + (int64_t)login.expires_in * 1000000;
    portENTER_CRITICAL(&state_lock);
    state.state = DEVICE_LOGIN_WAITING;
    strcpy(state.user_code, login.user_code);
    strcpy(state.verification_uri, login.verification_uri);
    strcpy(state.verification_uri_complete, login.verification_uri_complete);
    snprintf(state.message,sizeof(state.message),"%s",node_setup ? "Approve remote control setup in NyxID." : "Open NyxID and select an existing Key or create a Key for this panel.");
    deadline_us = expires_at;
    portEXIT_CRITICAL(&state_lock);
    uint32_t interval = login.interval;
    while (!atomic_load(&cancelled)) {
        if (!wait_seconds(interval)) break;
        if (esp_timer_get_time() >= expires_at) { failure = "This code expired. Start again for a new code."; break; }
        body = cJSON_CreateObject();
        if (!body) break;
        cJSON_AddStringToObject(body, "device_code", login.device_code);
        status = request(HTTP_METHOD_POST, node_setup ? "/auth/device/poll" : "/auth/agent-key/poll", NULL, body, response);
        cJSON_Delete(body);
        device_login_poll_t poll = node_setup ? device_login_parse_poll(status, response) : device_agent_parse_poll(status, response);
        if (poll == LOGIN_POLL_APPROVED) {
            cJSON *json = cJSON_Parse(response);
            if (!node_setup) {
                nyx_agent_credential_t agent={0};
                const cJSON *key=cJSON_GetObjectItemCaseSensitive(json,"api_key");
                bool ok=json_copy(json,"credential",agent.credential,sizeof(agent.credential)) &&
                    json_copy(json,"credential_id",agent.credential_id,sizeof(agent.credential_id)) &&
                    json_copy(key,"id",agent.key_id,sizeof(agent.key_id)) &&
                    json_copy(key,"owner_id",agent.owner_id,sizeof(agent.owner_id));
                (void)json_copy(json,"label",agent.label,sizeof(agent.label));
                const cJSON *secret=cJSON_GetObjectItemCaseSensitive(json,"credential");
                if(cJSON_IsString(secret))erase(secret->valuestring,strlen(secret->valuestring));
                cJSON_Delete(json);
                if (ok && !atomic_load(&cancelled)) ready=nyx_creds_store_agent(&agent)==0;
                erase(&agent,sizeof(agent));
                failure="Could not save the selected Key. Please sign in again.";
                goto done;
            }
            const cJSON *value = cJSON_GetObjectItemCaseSensitive(json, "access_token");
            if (cJSON_IsString(value) && strlen(value->valuestring) <= 8192) access = strdup(value->valuestring);
            if (cJSON_IsString(value)) erase(value->valuestring, strlen(value->valuestring));
            value = cJSON_GetObjectItemCaseSensitive(json, "refresh_token");
            if (cJSON_IsString(value)) erase(value->valuestring, strlen(value->valuestring));
            cJSON_Delete(json);
            break;
        }
        if (poll == LOGIN_POLL_PENDING) continue;
        if (!status || status >= 500 || poll == LOGIN_POLL_RATE_LIMITED || poll == LOGIN_POLL_SLOW_DOWN) {
            interval = device_login_backoff(interval);
            continue;
        }
        failure = poll == LOGIN_POLL_DENIED ? "Sign-in was declined. You can start again." :
            poll == LOGIN_POLL_EXPIRED ? "This code expired. Start again for a new code." :
            poll == LOGIN_POLL_DELIVERED ? "Sign-in delivery was lost. Please request a new code." :
            "NyxID could not complete sign-in. Please start again.";
        break;
    }
    if (!access || atomic_load(&cancelled)) goto done;
    set_state(DEVICE_LOGIN_REGISTERING, "Registering this panel with NyxID...");
    failure = "Device setup is unfinished. Sign in again to resume safely.";
    if (!setup_node(access, response, &binding) || atomic_load(&cancelled)) goto done;
    set_state(DEVICE_LOGIN_SERVICE, "Setting up remote access...");
    if (!setup_service(access, response, &binding) || atomic_load(&cancelled)) goto done;
    if (nyx_creds_store_binding(&binding) != 0) { failure = "Could not save the connection. Please try again."; goto done; }
    nvs_handle_t h;
    if (nvs_open(NYX_NODE_SETUP_NVS_NAMESPACE, NVS_READWRITE, &h) == ESP_OK) {
        nvs_erase_key(h, "pending_v1"); nvs_commit(h); nvs_close(h);
    }
    ready = true;
done:
    if (access && response) {
        // Revoke only this temporary account session; never store it in NVS.
        for (int attempt = 0; attempt < 2; ++attempt) {
            int logout = request(HTTP_METHOD_POST, "/auth/logout", access, NULL, response);
            if ((logout >= 200 && logout < 300) || logout == 401) break;
        }
        erase(access, strlen(access)); free(access);
    }
    if (response) { erase(response, RESPONSE_CAP); free(response); }
    erase(&login, sizeof(login)); erase(&binding, sizeof(binding));
    set_state(ready ? DEVICE_LOGIN_READY : atomic_load(&cancelled) ? DEVICE_LOGIN_CANCELLED : DEVICE_LOGIN_FAILED,
        ready ? (node_setup ? "Connection saved. Restart the panel to activate it." : "Key saved. Refresh the Bot list and select a Bot for voice.") :
        atomic_load(&cancelled) ? "Sign-in cancelled." : failure);
    portENTER_CRITICAL(&state_lock);
    state.busy = false;
    portEXIT_CRITICAL(&state_lock);
    vTaskDelete(NULL);
}

static esp_err_t start_login(bool node_setup)
{
    bool online = net_manager_connected();
    portENTER_CRITICAL(&state_lock);
    if (state.busy) { portEXIT_CRITICAL(&state_lock); return ESP_ERR_INVALID_STATE; }
    memset(&state, 0, sizeof(state));
    state.node_setup = node_setup;
    deadline_us = 0;
    if (!online) {
        state.state = DEVICE_LOGIN_FAILED;
        strcpy(state.message, "Connect to Wi-Fi in Settings before signing in.");
        portEXIT_CRITICAL(&state_lock);
        return ESP_ERR_INVALID_STATE;
    }
    state.busy = true;
    state.state = DEVICE_LOGIN_REQUESTING;
    strcpy(state.message, "Getting a sign-in code...");
    atomic_store(&cancelled, false);
    portEXIT_CRITICAL(&state_lock);
    if (xTaskCreate(login_task, "nyx_setup", 12288, (void *)(uintptr_t)node_setup, 4, NULL) != pdPASS) {
        set_state(DEVICE_LOGIN_FAILED, "Not enough memory to start sign-in.");
        portENTER_CRITICAL(&state_lock); state.busy = false; portEXIT_CRITICAL(&state_lock);
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t device_code_start_if_needed(void) { return start_login(false); }
esp_err_t device_code_start_node_setup(void) { return start_login(true); }
