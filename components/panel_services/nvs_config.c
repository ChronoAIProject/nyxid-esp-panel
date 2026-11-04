#include "nvs_config.h"

#include "devices.h"
#include "settings.h"

#include "nvs.h"
#include "nvs_flash.h"
#include "esp_log.h"

#include <stdio.h>
#include <string.h>

static const char *TAG = "haCam.nvs";

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (cap > 0u) {
        snprintf(dst, cap, "%s", src ? src : "");
    }
}

static void get_str_default(nvs_handle_t h, const char *key, char *dst, size_t cap)
{
    size_t len = cap;
    if (nvs_get_str(h, key, dst, &len) != ESP_OK) {
        return;
    }
    if (cap > 0u) {
        dst[cap - 1u] = '\0';
    }
}

void nvs_config_defaults_stream(cfg_stream_t *o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    o->fps = VIDEO_FPS;
    o->bitrate = VIDEO_BITRATE;
}

void nvs_config_defaults_lcd(cfg_lcd_t *o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    copy_str(o->idle_mode, sizeof(o->idle_mode), "qr");
    o->brightness = 80u;
    o->rotation = 0u;
}

void nvs_config_defaults_device(cfg_device_t *o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    const device_config_t *self = devices_self();
    copy_str(o->label, sizeof(o->label),
             self && self->label ? self->label : HA_CAM_DEFAULT_LABEL);
    if (self && self->site) {
        copy_str(o->location, sizeof(o->location), self->site);
    }
}

void nvs_config_defaults_wifi(cfg_wifi_t *o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    const device_config_t *self = devices_self();
    if (!self) return;
    copy_str(o->ssid_1, sizeof(o->ssid_1), self->wifi_ssid_1);
    copy_str(o->psw_1,  sizeof(o->psw_1),  self->wifi_password_1);
    copy_str(o->ssid_2, sizeof(o->ssid_2), self->wifi_ssid_2);
    copy_str(o->psw_2,  sizeof(o->psw_2),  self->wifi_password_2);
}

void nvs_config_defaults_lan(cfg_lan_t *o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    o->enabled = true;
    o->port = 80u;
}

void nvs_config_defaults_mqtt(cfg_mqtt_t *o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    o->enabled = true;
    const device_config_t *self = devices_self();
    if (self) {
        copy_str(o->host, sizeof(o->host), self->mqtt_host);
        o->port = self->mqtt_port ? self->mqtt_port : 1883u;
        copy_str(o->user, sizeof(o->user), self->mqtt_user);
        copy_str(o->pass, sizeof(o->pass), self->mqtt_pass);
        snprintf(o->base_topic, sizeof(o->base_topic), "haCam/%s",
                 self->slug && self->slug[0] ? self->slug : "unknown");
    } else {
        o->port = 1883u;
    }
}

esp_err_t nvs_config_get_stream(cfg_stream_t *o)
{
    if (!o) return ESP_ERR_INVALID_ARG;
    nvs_config_defaults_stream(o);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("stream", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return ESP_OK;
    if (err != ESP_OK) return err;
    get_str_default(h, "target", o->target_whip_url, sizeof(o->target_whip_url));
    uint8_t auto_start = o->auto_start ? 1u : 0u;
    if (nvs_get_u8(h, "auto", &auto_start) == ESP_OK) o->auto_start = auto_start != 0u;
    (void)nvs_get_u8(h, "fps", &o->fps);
    (void)nvs_get_u32(h, "bitrate", &o->bitrate);
    nvs_close(h);
    if (o->fps == 0u) {
        o->fps = VIDEO_FPS;
    }
    if (o->bitrate == 0u) {
        o->bitrate = VIDEO_BITRATE;
    }
    return ESP_OK;
}

esp_err_t nvs_config_set_stream(const cfg_stream_t *i)
{
    if (!i) return ESP_ERR_INVALID_ARG;
    ESP_LOGI(TAG, "write namespace=stream keys=target,auto,fps,bitrate target=%s auto=%d fps=%u bitrate=%lu",
             i->target_whip_url, i->auto_start ? 1 : 0, (unsigned)i->fps,
             (unsigned long)i->bitrate);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("stream", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_str(h, "target", i->target_whip_url);
    if (err == ESP_OK) err = nvs_set_u8(h, "auto", i->auto_start ? 1u : 0u);
    if (err == ESP_OK) err = nvs_set_u8(h, "fps", i->fps);
    if (err == ESP_OK) err = nvs_set_u32(h, "bitrate", i->bitrate);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_get_lcd(cfg_lcd_t *o)
{
    if (!o) return ESP_ERR_INVALID_ARG;
    nvs_config_defaults_lcd(o);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("lcd", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return ESP_OK;
    if (err != ESP_OK) return err;
    get_str_default(h, "idle_mode", o->idle_mode, sizeof(o->idle_mode));
    (void)nvs_get_u8(h, "brightness", &o->brightness);
    (void)nvs_get_u8(h, "rotation", &o->rotation);
    nvs_close(h);
    return ESP_OK;
}

esp_err_t nvs_config_set_lcd(const cfg_lcd_t *i)
{
    if (!i) return ESP_ERR_INVALID_ARG;
    ESP_LOGI(TAG, "write namespace=lcd keys=idle_mode,brightness,rotation idle_mode=%s brightness=%u rotation=%u",
             i->idle_mode, (unsigned)i->brightness, (unsigned)i->rotation);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("lcd", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_str(h, "idle_mode", i->idle_mode);
    if (err == ESP_OK) err = nvs_set_u8(h, "brightness", i->brightness);
    if (err == ESP_OK) err = nvs_set_u8(h, "rotation", i->rotation);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_get_device(cfg_device_t *o)
{
    if (!o) return ESP_ERR_INVALID_ARG;
    nvs_config_defaults_device(o);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("dev", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return ESP_OK;
    if (err != ESP_OK) return err;
    get_str_default(h, "label", o->label, sizeof(o->label));
    get_str_default(h, "location", o->location, sizeof(o->location));
    nvs_close(h);
    return ESP_OK;
}

esp_err_t nvs_config_set_device(const cfg_device_t *i)
{
    if (!i) return ESP_ERR_INVALID_ARG;
    ESP_LOGI(TAG, "write namespace=dev keys=label,location label=%s location=%s",
             i->label, i->location);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("dev", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_str(h, "label", i->label);
    if (err == ESP_OK) err = nvs_set_str(h, "location", i->location);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_get_wifi(cfg_wifi_t *o)
{
    if (!o) return ESP_ERR_INVALID_ARG;
    nvs_config_defaults_wifi(o);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("wifi", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return ESP_OK;
    if (err != ESP_OK) return err;
    get_str_default(h, "ssid_1", o->ssid_1, sizeof(o->ssid_1));
    get_str_default(h, "psw_1", o->psw_1, sizeof(o->psw_1));
    get_str_default(h, "ssid_2", o->ssid_2, sizeof(o->ssid_2));
    get_str_default(h, "psw_2", o->psw_2, sizeof(o->psw_2));
    nvs_close(h);
    return ESP_OK;
}

esp_err_t nvs_config_set_wifi(const cfg_wifi_t *i)
{
    if (!i) return ESP_ERR_INVALID_ARG;
    ESP_LOGI(TAG, "write namespace=wifi keys=ssid_1,psw_1,ssid_2,psw_2 ssid_1=%s psw_1=<redacted> ssid_2=%s psw_2=<redacted>",
             i->ssid_1, i->ssid_2);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("wifi", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_str(h, "ssid_1", i->ssid_1);
    if (err == ESP_OK) err = nvs_set_str(h, "psw_1", i->psw_1);
    if (err == ESP_OK) err = nvs_set_str(h, "ssid_2", i->ssid_2);
    if (err == ESP_OK) err = nvs_set_str(h, "psw_2", i->psw_2);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_get_lan(cfg_lan_t *o)
{
    if (!o) return ESP_ERR_INVALID_ARG;
    nvs_config_defaults_lan(o);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("lan", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return ESP_OK;
    if (err != ESP_OK) return err;
    uint8_t enabled = o->enabled ? 1u : 0u;
    if (nvs_get_u8(h, "enabled", &enabled) == ESP_OK) o->enabled = enabled != 0u;
    (void)nvs_get_u16(h, "port", &o->port);
    if (o->port == 0u) o->port = 80u;
    nvs_close(h);
    return ESP_OK;
}

esp_err_t nvs_config_set_lan(const cfg_lan_t *i)
{
    if (!i || i->port == 0u) return ESP_ERR_INVALID_ARG;
    ESP_LOGI(TAG, "write namespace=lan keys=enabled,port enabled=%d port=%u",
             i->enabled ? 1 : 0, (unsigned)i->port);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("lan", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_u8(h, "enabled", i->enabled ? 1u : 0u);
    if (err == ESP_OK) err = nvs_set_u16(h, "port", i->port);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_get_mqtt(cfg_mqtt_t *o)
{
    if (!o) return ESP_ERR_INVALID_ARG;
    nvs_config_defaults_mqtt(o);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("mqtt", NVS_READONLY, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) return ESP_OK;
    if (err != ESP_OK) return err;
    uint8_t enabled = o->enabled ? 1u : 0u;
    if (nvs_get_u8(h, "enabled", &enabled) == ESP_OK) o->enabled = enabled != 0u;
    get_str_default(h, "host", o->host, sizeof(o->host));
    (void)nvs_get_u16(h, "port", &o->port);
    get_str_default(h, "user", o->user, sizeof(o->user));
    get_str_default(h, "pass", o->pass, sizeof(o->pass));
    get_str_default(h, "base_topic", o->base_topic, sizeof(o->base_topic));
    if (o->port == 0u) o->port = 1883u;
    if (o->host[0] == '\0') {
        const device_config_t *self = devices_self();
        copy_str(o->host, sizeof(o->host), self ? self->mqtt_host : "");
    }
    if (o->base_topic[0] == '\0') {
        const device_config_t *self = devices_self();
        if (self && self->slug && self->slug[0]) {
            snprintf(o->base_topic, sizeof(o->base_topic), "haCam/%s", self->slug);
        }
    }
    nvs_close(h);
    return ESP_OK;
}

esp_err_t nvs_config_set_mqtt(const cfg_mqtt_t *i)
{
    if (!i || i->port == 0u ||
        (i->enabled && (i->host[0] == '\0' || i->base_topic[0] == '\0'))) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_LOGI(TAG, "write namespace=mqtt keys=enabled,host,port,user,pass,base_topic enabled=%d host=%s port=%u user=%s pass=<redacted> base_topic=%s",
             i->enabled ? 1 : 0, i->host, (unsigned)i->port, i->user, i->base_topic);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("mqtt", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_u8(h, "enabled", i->enabled ? 1u : 0u);
    if (err == ESP_OK) err = nvs_set_str(h, "host", i->host);
    if (err == ESP_OK) err = nvs_set_u16(h, "port", i->port);
    if (err == ESP_OK) err = nvs_set_str(h, "user", i->user);
    if (err == ESP_OK) err = nvs_set_str(h, "pass", i->pass);
    if (err == ESP_OK) err = nvs_set_str(h, "base_topic", i->base_topic);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_set_legacy_wifi_primary(const cfg_wifi_t *i)
{
    if (!i) return ESP_ERR_INVALID_ARG;
    ESP_LOGI(TAG, "write namespace=wifi-set keys=ssid,psw ssid=%s psw=<redacted>", i->ssid_1);
    nvs_handle_t h = 0;
    esp_err_t err = nvs_open("wifi-set", NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    err = nvs_set_str(h, "ssid", i->ssid_1);
    if (err == ESP_OK) err = nvs_set_str(h, "psw", i->psw_1);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}

esp_err_t nvs_config_factory_reset_preserve(void)
{
    // Erase configuration in place. Erasing the partition and restoring only the
    // active identity loses inactive server profiles and risks credentials on a
    // failed restore. Wi-Fi, every identity profile and pending setup stay intact.
    static const char *const reset_namespaces[] = {
        "stream", "lcd", "dev", "lan", "mqtt", "panel_art",
    };
    ESP_LOGW(TAG, "factory reset configuration; preserve wifi and all identities");
    for (size_t i = 0; i < sizeof(reset_namespaces) / sizeof(reset_namespaces[0]); ++i) {
        nvs_handle_t h = 0;
        esp_err_t err = nvs_open(reset_namespaces[i], NVS_READONLY, &h);
        if (err == ESP_ERR_NVS_NOT_FOUND) continue;
        if (err != ESP_OK) return err;
        nvs_close(h);
        err = nvs_open(reset_namespaces[i], NVS_READWRITE, &h);
        if (err != ESP_OK) return err;
        err = nvs_erase_all(h);
        if (err == ESP_OK) err = nvs_commit(h);
        nvs_close(h);
        if (err != ESP_OK) return err;
    }
    return ESP_OK;
}

esp_err_t nvs_config_factory_reset_all(void)
{
    ESP_LOGW(TAG, "factory reset all namespaces");
    nvs_flash_deinit();
    esp_err_t err = nvs_flash_erase();
    if (err == ESP_OK) {
        err = nvs_flash_init();
    }
    return err;
}
