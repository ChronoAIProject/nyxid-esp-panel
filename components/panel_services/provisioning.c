#include "provisioning.h"

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "haCam.prov";

static struct {
    SemaphoreHandle_t lock;
    bool active;
    uint8_t consecutive_wifi_failures;
} g;

static void ensure_lock(void)
{
    if (!g.lock) {
        g.lock = xSemaphoreCreateMutex();
    }
}

bool provisioning_mode_active(void)
{
    bool active = false;
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
        active = g.active;
        xSemaphoreGive(g.lock);
    }
    return active;
}

void provisioning_mode_enter(const char *reason)
{
    bool should_log = false;
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
        should_log = !g.active;
        g.active = true;
        xSemaphoreGive(g.lock);
    }
    if (should_log) {
        ESP_LOGW(TAG, "Entering provisioning mode reason=%s wifi_failures=%u",
                 reason && reason[0] ? reason : "", g.consecutive_wifi_failures);
        ESP_LOGW(TAG, "Entering provisioning mode%s%s",
                 reason && reason[0] ? ": " : "",
                 reason && reason[0] ? reason : "");
    }
}

void provisioning_mode_exit(void)
{
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
        g.active = false;
        g.consecutive_wifi_failures = 0u;
        xSemaphoreGive(g.lock);
    }
    ESP_LOGI(TAG, "Exited provisioning mode reason=manual_or_success wifi_failures=0");
}

void provisioning_mode_note_wifi_failure(const net_manager_snapshot_t *snapshot)
{
    if (snapshot && (snapshot->eth_link || snapshot->eth_ip)) {
        return;
    }

    uint8_t failures = 0u;
    bool enter = false;
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
        if (!g.active && g.consecutive_wifi_failures < UINT8_MAX) {
            g.consecutive_wifi_failures++;
        }
        failures = g.consecutive_wifi_failures;
        enter = !g.active && failures >= 3u;
        xSemaphoreGive(g.lock);
    }

    ESP_LOGW(TAG, "WiFi failure count=%u", failures);
    if (enter) {
        provisioning_mode_enter("persistent WiFi failure");
    }
}

void provisioning_mode_note_network_up(void)
{
    ensure_lock();
    if (g.lock) {
        xSemaphoreTake(g.lock, portMAX_DELAY);
        g.consecutive_wifi_failures = 0u;
        xSemaphoreGive(g.lock);
    }
    ESP_LOGD(TAG, "network up observed; wifi failure counter reset");
}
