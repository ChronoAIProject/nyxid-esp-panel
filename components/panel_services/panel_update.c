#include "panel_update.h"
#include "panel_ota.h"
#include "panel_voice.h"
#include "media_mode.h"
#include "net_manager.h"
#include "panel_runtime.h"
#include "panel_flash.h"
#include "esp_app_desc.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_timer.h"
#include "nvs.h"
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#ifndef NYX_PANEL_UPDATE_DISABLED
#define NYX_PANEL_UPDATE_DISABLED 0
#endif
#if CONFIG_ESP32P4_REV_MIN_FULL < 300
#define UPDATE_PROFILE "p4-rev1-16mb"
#else
#define UPDATE_PROFILE "p4-rev3-16mb"
#endif
#define CHECK_INTERVAL_US (6LL*60*60*1000000)
static portMUX_TYPE lock=portMUX_INITIALIZER_UNLOCKED;
static panel_update_snapshot_t snapshot;
static atomic_int requested;
static atomic_bool started;
static const char *TAG="panel.update";

const char *panel_update_profile(void) { return UPDATE_PROFILE; }
void panel_update_snapshot(panel_update_snapshot_t *out)
{
    if (!out) return;
    portENTER_CRITICAL(&lock); *out=snapshot; portEXIT_CRITICAL(&lock);
}
static void state(const char *value,esp_err_t err)
{
    portENTER_CRITICAL(&lock);
    snprintf(snapshot.state,sizeof(snapshot.state),"%s",value);
    snapshot.last_error=err;
    portEXIT_CRITICAL(&lock);
}
static esp_err_t save_string(const char *key,const char *value)
{
    nvs_handle_t h;
    panel_flash_lock();
    esp_err_t err=nvs_open("panel_update",NVS_READWRITE,&h);
    if (err==ESP_OK) {
        err=nvs_set_str(h,key,value);
        if (err==ESP_OK) err=nvs_commit(h);
        nvs_close(h);
    }
    panel_flash_unlock();
    return err;
}
esp_err_t panel_update_record_pending(const char *version)
{
    return save_string("pending",version);
}
esp_err_t panel_update_set_automatic(bool enabled)
{
    if (!atomic_load(&started) || NYX_PANEL_UPDATE_DISABLED) return ESP_ERR_INVALID_STATE;
    nvs_handle_t h;
    panel_flash_lock();
    esp_err_t err=nvs_open("panel_update",NVS_READWRITE,&h);
    if (err==ESP_OK) {
        err=nvs_set_u8(h,"automatic",enabled);
        if (err==ESP_OK) err=nvs_commit(h);
        nvs_close(h);
    }
    panel_flash_unlock();
    if (err==ESP_OK) {
        portENTER_CRITICAL(&lock); snapshot.automatic=enabled; portEXIT_CRITICAL(&lock);
    }
    return err;
}
esp_err_t panel_update_check(bool install)
{
    if (!atomic_load(&started) || NYX_PANEL_UPDATE_DISABLED || panel_ota_busy()) return ESP_ERR_INVALID_STATE;
    panel_update_snapshot_t s;panel_update_snapshot(&s);
    if (s.checking) return ESP_ERR_INVALID_STATE;
    int expected=0;
    return atomic_compare_exchange_strong(&requested,&expected,install ? 2 : 1) ? ESP_OK : ESP_ERR_INVALID_STATE;
}
static bool idle(void)
{
    panel_voice_snapshot_t voice; panel_voice_snapshot(&voice);
    return !voice.busy && !voice.active && media_mode_current()==MEDIA_MODE_IDLE && !panel_ota_busy();
}
static bool boot_valid(void)
{
    esp_ota_img_states_t boot;
    panel_flash_lock();
    const esp_partition_t *running=esp_ota_get_running_partition();
    esp_err_t err=running ? esp_ota_get_state_partition(running,&boot) : ESP_ERR_NOT_FOUND;
    panel_flash_unlock();
    return running && (err==ESP_ERR_NOT_FOUND || (err==ESP_OK && boot==ESP_OTA_IMG_VALID));
}
static esp_err_t fetch_release(panel_release_t *release)
{
    char url[256];
    snprintf(url,sizeof(url),PANEL_RELEASE_ROOT "latest/download/update-%s.json",UPDATE_PROFILE);
    esp_http_client_config_t config={.url=url,.crt_bundle_attach=esp_crt_bundle_attach,
        .timeout_ms=15000,.disable_auto_redirect=true,.buffer_size=2048,
        // GitHub asset redirects carry a signed query longer than IDF's
        // default 512-byte request-line buffer.
        .buffer_size_tx=4096};
    esp_http_client_handle_t http=esp_http_client_init(&config);
    if (!http) return ESP_ERR_NO_MEM;
    esp_http_client_set_header(http,"User-Agent","nyxid-esp-panel-updater");
    esp_err_t err=ESP_OK;
    int64_t length=0;
    for (unsigned redirects=0;err==ESP_OK;++redirects) {
        err=esp_http_client_open(http,0);
        if (err!=ESP_OK) break;
        length=esp_http_client_fetch_headers(http);
        int code=esp_http_client_get_status_code(http);
        ESP_LOGI(TAG,"Release metadata HTTP status=%d bytes=%lld",code,(long long)length);
        if (code==200) break;
        if (redirects>=3 || (code!=301 && code!=302 && code!=303 && code!=307 && code!=308)) {
            err=ESP_ERR_INVALID_RESPONSE;break;
        }
        err=esp_http_client_set_redirection(http);
        if (err==ESP_OK) {
            char next[2049];err=esp_http_client_get_url(http,next,sizeof(next));
            if (err==ESP_OK && !panel_update_download_url_allowed(next)) err=ESP_ERR_INVALID_ARG;
        }
        esp_http_client_close(http);
    }
    char body[4097];size_t received=0;
    if (err==ESP_OK && (length<=0 || length>4096)) err=ESP_ERR_INVALID_SIZE;
    while (err==ESP_OK && received<(size_t)length) {
        int count=esp_http_client_read(http,body+received,(size_t)length-received);
        if (count<=0) { err=ESP_ERR_TIMEOUT;break; }
        received+=(size_t)count;
    }
    esp_http_client_cleanup(http);
    if (err==ESP_OK && !panel_update_parse_release(body,received,UPDATE_PROFILE,release)) err=ESP_ERR_INVALID_RESPONSE;
    return err;
}
static void update_task(void *arg)
{
    // Do not delay the Panel or race the 30-second boot-health decision.
    vTaskDelay(pdMS_TO_TICKS(60000));
    nvs_handle_t h;char pending[32]="";
    panel_flash_lock();
    if (nvs_open("panel_update",NVS_READONLY,&h)==ESP_OK) {
        size_t cap=sizeof(pending);nvs_get_str(h,"pending",pending,&cap);nvs_close(h);
    }
    panel_flash_unlock();
    while (pending[0]) {
        esp_err_t err=ESP_OK;
        if (strcmp(pending,esp_app_get_description()->version)) {
            err=save_string("blocked",pending);
            if (err==ESP_OK) {
                portENTER_CRITICAL(&lock);
                snprintf(snapshot.blocked_version,sizeof(snapshot.blocked_version),"%s",pending);
                portEXIT_CRITICAL(&lock);
                state("rolled_back",ESP_FAIL);
            }
        }
        if (err==ESP_OK) err=save_string("pending","");
        if (err==ESP_OK) break;
        state("storage_error",err);
        // Keep the durable target until rollback suppression has been saved.
        vTaskDelay(pdMS_TO_TICKS(15000));
    }
    int64_t next_check=0;
    panel_release_t release={0};bool want_install=false,manual_install=false;
    for (;;) {
        panel_update_snapshot_t s;panel_update_snapshot(&s);
        int request=atomic_exchange(&requested,0);
        bool online=net_manager_connected() && panel_runtime_ready() && boot_valid();
        if (request && !online) state("offline",ESP_ERR_INVALID_STATE);
        bool due=s.automatic && esp_timer_get_time()>=next_check;
        if (online && (request || due) && !panel_ota_busy()) {
            // Even metadata TLS allocation should wait until microphone/camera are idle.
            if (!idle()) {
                state("waiting_for_idle",ESP_OK);
                if (request) { int none=0; atomic_compare_exchange_strong(&requested,&none,request); }
            } else {
                portENTER_CRITICAL(&lock);snapshot.checking=true;portEXIT_CRITICAL(&lock);
                state("checking",ESP_OK);
                panel_release_t found={0};esp_err_t err=fetch_release(&found);
                bool available=err==ESP_OK && panel_update_version_newer(found.version,esp_app_get_description()->version);
                portENTER_CRITICAL(&lock);
                snapshot.checking=false;snapshot.available=available;
                if (err==ESP_OK) snprintf(snapshot.latest_version,sizeof(snapshot.latest_version),"%s",found.version);
                portEXIT_CRITICAL(&lock);
                next_check=esp_timer_get_time()+(err==ESP_OK ? CHECK_INTERVAL_US : 15LL*60*1000000);
                manual_install=request==2;
                want_install=available && (manual_install || (!request && s.automatic));
                release=found;
                state(err!=ESP_OK ? "check_failed" : available ? "update_available" : "up_to_date",err);
                ESP_LOGI(TAG,"Release check: %s",esp_err_to_name(err));
            }
        }
        panel_update_snapshot(&s);
        if (want_install && !s.automatic && !manual_install) want_install=false;
        if (!strcmp(s.state,"installing") && !panel_ota_busy() && panel_ota_error()!=ESP_OK) {
            state("install_failed",panel_ota_error());
            next_check=esp_timer_get_time()+15LL*60*1000000;
        }
        if (want_install && online && idle()) {
            want_install=false;
            if (!strcmp(release.version,s.blocked_version)) state("rolled_back",ESP_ERR_INVALID_VERSION);
            else {
                esp_err_t err=panel_ota_start_url(release.url,release.sha256,release.version,release.size);
                if (err==ESP_ERR_INVALID_STATE) {want_install=true;state("waiting_for_idle",ESP_OK);}
                else state(err==ESP_OK ? "installing" : "install_failed",err);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
void panel_update_init(void)
{
    if (atomic_exchange(&started,true)) return;
    snapshot.automatic=!NYX_PANEL_UPDATE_DISABLED;
    snprintf(snapshot.state,sizeof(snapshot.state),"idle");
    nvs_handle_t h;
    if (nvs_open("panel_update",NVS_READONLY,&h)==ESP_OK) {
        uint8_t enabled;
        if (nvs_get_u8(h,"automatic",&enabled)==ESP_OK) snapshot.automatic=enabled!=0;
        size_t cap=sizeof(snapshot.blocked_version);
        nvs_get_str(h,"blocked",snapshot.blocked_version,&cap);
        nvs_close(h);
    }
    if (NYX_PANEL_UPDATE_DISABLED) {
        snapshot.automatic=false;state("development",ESP_OK);return;
    }
    if (xTaskCreate(update_task,"panel_update",12*1024,NULL,3,NULL)!=pdPASS) {
        atomic_store(&started,false);state("unavailable",ESP_ERR_NO_MEM);
    }
}
