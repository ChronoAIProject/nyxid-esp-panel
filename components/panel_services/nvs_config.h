#pragma once

#include "esp_err.h"

#include <stdbool.h>
#include <stdint.h>

#define CFG_STREAM_TARGET_MAX 256u
#define CFG_LCD_MODE_MAX      16u
#define CFG_DEVICE_TEXT_MAX   48u
#define CFG_WIFI_SSID_MAX     33u
#define CFG_WIFI_PSW_MAX      64u
#define CFG_MQTT_HOST_MAX     64u
#define CFG_MQTT_USER_MAX     32u
#define CFG_MQTT_PASS_MAX     64u
#define CFG_MQTT_TOPIC_MAX    64u

typedef struct {
    char target_whip_url[CFG_STREAM_TARGET_MAX];
    bool auto_start;
    uint8_t fps;
    uint32_t bitrate;
} cfg_stream_t;

typedef struct {
    char idle_mode[CFG_LCD_MODE_MAX];
    uint8_t brightness;
    uint8_t rotation;
} cfg_lcd_t;

typedef struct {
    char label[CFG_DEVICE_TEXT_MAX];
    char location[CFG_DEVICE_TEXT_MAX];
} cfg_device_t;

typedef struct {
    char ssid_1[CFG_WIFI_SSID_MAX];
    char psw_1[CFG_WIFI_PSW_MAX];
    char ssid_2[CFG_WIFI_SSID_MAX];
    char psw_2[CFG_WIFI_PSW_MAX];
} cfg_wifi_t;

typedef struct {
    bool enabled;
    uint16_t port;
} cfg_lan_t;

typedef struct {
    bool enabled;
    char host[CFG_MQTT_HOST_MAX];
    uint16_t port;
    char user[CFG_MQTT_USER_MAX];
    char pass[CFG_MQTT_PASS_MAX];
    char base_topic[CFG_MQTT_TOPIC_MAX];
} cfg_mqtt_t;

void nvs_config_defaults_stream(cfg_stream_t *o);
void nvs_config_defaults_lcd(cfg_lcd_t *o);
void nvs_config_defaults_device(cfg_device_t *o);
void nvs_config_defaults_wifi(cfg_wifi_t *o);
void nvs_config_defaults_lan(cfg_lan_t *o);
void nvs_config_defaults_mqtt(cfg_mqtt_t *o);

esp_err_t nvs_config_get_stream(cfg_stream_t *o);
esp_err_t nvs_config_set_stream(const cfg_stream_t *i);
esp_err_t nvs_config_get_lcd(cfg_lcd_t *o);
esp_err_t nvs_config_set_lcd(const cfg_lcd_t *i);
esp_err_t nvs_config_get_device(cfg_device_t *o);
esp_err_t nvs_config_set_device(const cfg_device_t *i);
esp_err_t nvs_config_get_wifi(cfg_wifi_t *o);
esp_err_t nvs_config_set_wifi(const cfg_wifi_t *i);
esp_err_t nvs_config_get_lan(cfg_lan_t *o);
esp_err_t nvs_config_set_lan(const cfg_lan_t *i);
esp_err_t nvs_config_get_mqtt(cfg_mqtt_t *o);
esp_err_t nvs_config_set_mqtt(const cfg_mqtt_t *i);
esp_err_t nvs_config_set_legacy_wifi_primary(const cfg_wifi_t *i);
esp_err_t nvs_config_factory_reset_preserve(void);
esp_err_t nvs_config_factory_reset_all(void);
