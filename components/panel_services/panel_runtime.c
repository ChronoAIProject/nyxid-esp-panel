/* voice_presence_ha_camera
 *
 * ESP32-P4 HA camera node firmware.
 */

#include <esp_event.h>
#include <esp_log.h>
#include <esp_err.h>
#include <esp_app_desc.h>
#include <esp_system.h>
#include <esp_sntp.h>
#include <nvs_flash.h>
#include <nvs.h>
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_console.h"
#include "common.h"
#include "devices.h"
#include "settings.h"
#include "bsp/esp-bsp.h"
#include "device_code.h"
#include "lcd_renderer.h"
#include "media_mode.h"
#include "local_api.h"
#include "local_http_server.h"
#include "net_manager.h"
#include "nvs_config.h"
#include "nyx_creds.h"
#include "nyx_node.h"
#include "provisioning.h"
#include "qr_scanner.h"
#include "panel_runtime.h"
#include "panel_voice.h"
#include "product_support.h"
#include <stdatomic.h>

#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static const char *TAG = "haCam";

static void clock_synced(struct timeval *now)
{
    ESP_LOGI(TAG, "System clock synchronized");
}

static void start_clock_if_connected(void)
{
    // Signed Nyx requests need wall time even when Settings has never opened.
    // Only the service task initializes SNTP; the image page never waits for it.
    if (esp_sntp_enabled() || !network_is_connected()) return;
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "time.cloudflare.com");
#if CONFIG_LWIP_SNTP_MAX_SERVERS > 1
    esp_sntp_setservername(1, "pool.ntp.org");
#endif
    esp_sntp_set_time_sync_notification_cb(clock_synced);
    esp_sntp_init();
}

// --- Console commands ---

static int stop_cli(int argc, char **argv)
{
    panel_runtime_set_preview(false);
    return 0;
}

static int sys_cli(int argc, char **argv)
{
    sys_state_show();
    return 0;
}

static int wifi_cli(int argc, char **argv)
{
    if (argc < 2) return -1;
    cfg_wifi_t wifi = {0};
    nvs_config_defaults_wifi(&wifi);
    snprintf(wifi.ssid_1, sizeof(wifi.ssid_1), "%s", argv[1]);
    snprintf(wifi.psw_1, sizeof(wifi.psw_1), "%s", argc > 2 ? argv[2] : "");
    wifi.ssid_2[0] = '\0';
    wifi.psw_2[0] = '\0';
    (void)nvs_config_set_wifi(&wifi);
    (void)nvs_config_set_legacy_wifi_primary(&wifi);
    return network_connect_wifi(wifi.ssid_1, wifi.psw_1);
}

static int wifi_reconnect_cli(int argc, char **argv)
{
    (void)argc;
    (void)argv;
#ifdef CONFIG_NETWORK_USE_ETHERNET
    return ESP_ERR_NOT_SUPPORTED;
#else
    // The shared event handler reconnects using the station's saved settings.
    return esp_wifi_disconnect();
#endif
}

static bool is_uuid_shape(const char *text)
{
    static const int dash_pos[] = {8, 13, 18, 23};
    if (!text || strlen(text) != 36u) {
        return false;
    }
    for (int i = 0; i < 36; ++i) {
        bool dash = false;
        for (size_t j = 0; j < sizeof(dash_pos) / sizeof(dash_pos[0]); ++j) {
            dash = dash || (i == dash_pos[j]);
        }
        if (dash) {
            if (text[i] != '-') return false;
        } else if (!isxdigit((unsigned char)text[i])) {
            return false;
        }
    }
    return true;
}

static const char *nyx_link_text(void)
{
    switch (nyx_node_state()) {
    case NYX_NODE_UP:
        return "up";
    case NYX_NODE_CONNECTING:
        return "connecting";
    case NYX_NODE_DOWN:
    default:
        return "down";
    }
}

static void read_legacy_wifi(char *ssid, size_t ssid_cap, char *psw, size_t psw_cap)
{
    if (!ssid || ssid_cap == 0u) {
        return;
    }
    ssid[0] = '\0';
    if (psw && psw_cap > 0u) {
        psw[0] = '\0';
    }
    nvs_handle_t h = 0;
    if (nvs_open("wifi-set", NVS_READONLY, &h) == ESP_OK) {
        size_t len = ssid_cap;
        if (nvs_get_str(h, "ssid", ssid, &len) == ESP_OK) {
            ssid[ssid_cap - 1u] = '\0';
        }
        if (psw && psw_cap > 0u) {
            len = psw_cap;
            if (nvs_get_str(h, "psw", psw, &len) == ESP_OK) {
                psw[psw_cap - 1u] = '\0';
            }
        }
        nvs_close(h);
    }
    if (ssid[0] == '\0') {
        cfg_wifi_t wifi = {0};
        if (nvs_config_get_wifi(&wifi) == ESP_OK) {
            snprintf(ssid, ssid_cap, "%s", wifi.ssid_1);
            if (psw && psw_cap > 0u) {
                snprintf(psw, psw_cap, "%s", wifi.psw_1);
            }
        }
    }
}

static int nyx_key_cli(int argc, char **argv)
{
    if (argc < 2 || argv[1][0] == '\0') {
        printf("Usage: nyx-key <auth_token>\n");
        return -1;
    }
    if (strncmp(argv[1], "nyx_nauth_", strlen("nyx_nauth_")) != 0) {
        printf("Warning: token does not start with nyx_nauth_; accepting anyway\n");
    }
    if (nyx_creds_save_api_key(argv[1]) != 0) {
        printf("Failed to save Nyx auth token\n");
        return -1;
    }
    printf("Set; reboot to apply\n");
    return 0;
}

static int nyx_node_cli(int argc, char **argv)
{
    if (argc < 2 || argv[1][0] == '\0') {
        printf("Usage: nyx-node <node_id>\n");
        return -1;
    }
    if (!is_uuid_shape(argv[1])) {
        printf("Invalid UUID shape; expected 8-4-4-4-12 hex\n");
        return -1;
    }
    if (nyx_creds_save_node_id(argv[1]) != 0) {
        printf("Failed to save Nyx node id\n");
        return -1;
    }
    printf("Set; reboot to apply\n");
    return 0;
}

static int nyx_secret_cli(int argc, char **argv)
{
    if (argc != 2 || strlen(argv[1]) != 64) {
        printf("Usage: nyx-secret <64-character signing_secret>\n"); return -1;
    }
    return nyx_creds_save_signing_secret(argv[1]);
}

static int nyx_slug_cli(int argc, char **argv)
{
    if (argc < 2 || argv[1][0] == '\0') {
        printf("Usage: nyx-slug <service_slug>\n");
        return -1;
    }
    if (nyx_creds_save_slug(argv[1]) != 0) {
        printf("Failed to save Nyx service slug\n");
        return -1;
    }
    printf("Set; reboot to apply\n");
    return 0;
}

static int nyx_status_cli(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    const char *api_key = nyx_creds_api_key();
    size_t api_len = api_key ? strlen(api_key) : 0u;
    char masked[32] = "";
    snprintf(masked, sizeof(masked), "%s", api_len ? "<set, redacted>" : "(unset)");

    net_manager_snapshot_t net = net_manager_snapshot();
    char ssid[CFG_WIFI_SSID_MAX] = "";
    read_legacy_wifi(ssid, sizeof(ssid), NULL, 0u);
    const char *ip = net.eth_ip ? net.eth_ip_addr : (net.wifi_ip ? net.wifi_ip_addr : "none");

    printf("api_key=%s  node_id=%s  slug=%s  fw=%s  wifi=%s  ip=%s  nyx_link=%s\n",
           masked,
           nyx_creds_node_id()[0] ? nyx_creds_node_id() : "(unset)",
           nyx_creds_service_slug()[0] ? nyx_creds_service_slug() : "(unset)",
           esp_app_get_description()->version,
           ssid[0] ? ssid : "(unset)",
           ip && ip[0] ? ip : "none",
           nyx_link_text());
    return 0;
}

static int screen_cli(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: screen <home|settings|wifi|device_bind|info|camera|scan|back>\n");
        return -1;
    }
    esp_err_t err = lcd_renderer_navigate(argv[1]);
    printf("Navigation: %s\n", esp_err_to_name(err));
    return err == ESP_OK ? 0 : -1;
}

static int nyx_login_cli(int argc, char **argv)
{
    if (argc != 2 || (strcmp(argv[1], "start") && strcmp(argv[1], "node-setup") && strcmp(argv[1], "status") && strcmp(argv[1], "cancel"))) {
        printf("Usage: nyx-login <start|node-setup|status|cancel>\n");
        return -1;
    }
    if (!strcmp(argv[1], "start") || !strcmp(argv[1], "node-setup")) {
        esp_err_t err = lcd_renderer_navigate("device_bind");
        if (err == ESP_OK) err = !strcmp(argv[1], "node-setup") ? device_code_start_node_setup() : device_code_start_if_needed();
        if (err != ESP_OK) { printf("Sign-in: %s\n", esp_err_to_name(err)); return -1; }
    } else if (!strcmp(argv[1], "cancel")) device_code_cancel();
    device_login_snapshot_t login;
    device_code_snapshot(&login);
    printf("Sign-in state=%d busy=%d remaining=%lu code=%s\n%s\n", login.state,
        login.busy, (unsigned long)login.seconds_remaining, login.user_code, login.message);
    return 0;
}

static int provision_mode_cli(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    provisioning_mode_enter("serial command");
    printf("Provisioning mode active\n");
    return 0;
}

static int exit_provision_cli(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    provisioning_mode_exit();
    char ssid[CFG_WIFI_SSID_MAX] = "";
    char psw[CFG_WIFI_PSW_MAX] = "";
    read_legacy_wifi(ssid, sizeof(ssid), psw, sizeof(psw));
    if (ssid[0] == '\0') {
        printf("Exited provisioning mode; no WiFi SSID configured\n");
        return 0;
    }
    network_connect_wifi(ssid, psw);
    printf("Exited provisioning mode; reconnecting to %s\n", ssid);
    return 0;
}

static int factory_reset_cli(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    esp_err_t err = nvs_config_factory_reset_all();
    if (err != ESP_OK) {
        printf("NVS wipe failed: %s\n", esp_err_to_name(err));
        return -1;
    }
    printf("Wiped. Reboot now (`reboot`) to enter provisioning mode.\n");
    return 0;
}

static int reboot_cli(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    esp_restart();
    return 0;
}

static esp_log_level_t parse_log_level(const char *text, bool *ok)
{
    if (ok) {
        *ok = true;
    }
    if (!text || text[0] == '\0') {
        if (ok) *ok = false;
        return ESP_LOG_NONE;
    }
    switch (text[0]) {
    case 'D':
    case 'd':
        return ESP_LOG_DEBUG;
    case 'I':
    case 'i':
        return ESP_LOG_INFO;
    case 'W':
    case 'w':
        return ESP_LOG_WARN;
    case 'E':
    case 'e':
        return ESP_LOG_ERROR;
    case 'N':
    case 'n':
        return ESP_LOG_NONE;
    default:
        if (ok) *ok = false;
        return ESP_LOG_NONE;
    }
}

static int loglevel_cli(int argc, char **argv)
{
    if (argc != 3) {
        printf("Usage: loglevel <tag> <D|I|W|E|N>\n");
        return -1;
    }
    bool ok = false;
    esp_log_level_t level = parse_log_level(argv[2], &ok);
    if (!ok) {
        printf("Usage: loglevel <tag> <D|I|W|E|N>\n");
        return -1;
    }
    esp_log_level_set(argv[1], level);
    ESP_LOGI(TAG, "runtime loglevel tag=%s level=%c", argv[1], argv[2][0]);
    printf("loglevel %s %c\n", argv[1], argv[2][0]);
    return 0;
}

static int nyx_voice_cli(int argc, char **argv)
{
    esp_err_t action=ESP_OK;
    if (argc==2 && !strcmp(argv[1],"refresh")) action=panel_voice_refresh_agents();
    else if (argc==3 && !strcmp(argv[1],"select")) action=panel_voice_configure(argv[2]);
    else if (argc==3 && !strcmp(argv[1],"gain")) {
        char *end=NULL;long gain=strtol(argv[2],&end,10);
        action=end && !*end && gain>=1 && gain<=8 ? panel_voice_gain_set((unsigned)gain) : ESP_ERR_INVALID_ARG;
    }
    else if (argc==3 && !strcmp(argv[1],"mode")) {
        action=!strcmp(argv[2],"hold") ? panel_voice_input_mode_set(PANEL_VOICE_INPUT_HOLD) :
            !strcmp(argv[2],"automatic") ? panel_voice_input_mode_set(PANEL_VOICE_INPUT_AUTOMATIC) : ESP_ERR_INVALID_ARG;
    }
    else if (argc==2 && !strcmp(argv[1],"mute")) panel_voice_microphone_mute(true);
    else if (argc==2 && !strcmp(argv[1],"unmute")) panel_voice_microphone_mute(false);
    else if (argc==2 && !strcmp(argv[1],"start")) action=panel_voice_start();
    else if (argc==2 && !strcmp(argv[1],"stop")) panel_voice_stop();
    else if (argc==2 && !strcmp(argv[1],"hold")) panel_voice_hold(true);
    else if (argc==2 && !strcmp(argv[1],"release")) panel_voice_hold(false);
    else if (argc==2 && !strcmp(argv[1],"list")) {
        panel_voice_bot_t *rows=calloc(PANEL_VOICE_MAX_BOTS,sizeof(*rows));
        if (!rows) return -1;
        size_t count=panel_voice_bots(rows,PANEL_VOICE_MAX_BOTS);
        for(size_t i=0;i<count;i++) printf("Voice Bot %s %s\n",rows[i].agent_id,rows[i].name);
        free(rows);
        return 0;
    } else if (!(argc==2 && (!strcmp(argv[1],"status") || !strcmp(argv[1],"peer-test")))) {
        printf("Usage: nyx-voice status|refresh|list|select <bot-id>|gain <1-8>|mode <hold|automatic>|mute|unmute|start|stop|hold|release|peer-test\n");
        return -1;
    }
    if (action!=ESP_OK) {
        printf("Voice action: %s\n",esp_err_to_name(action));
        return -1;
    }
    if (argc == 2 && strcmp(argv[1], "peer-test") == 0) {
        char result[256] = {0};
        esp_err_t err = panel_voice_peer_test(result, sizeof(result));
        printf("Voice peer test: %s %s\n", esp_err_to_name(err), result);
        return err == ESP_OK ? 0 : -1;
    }
    {
        panel_voice_snapshot_t snapshot = {0};
        panel_voice_snapshot(&snapshot);
        printf("Voice state=%s configured=%d busy=%d active=%d\n", snapshot.state,
               snapshot.configured, snapshot.busy, snapshot.active);
        printf("Voice audio sent=%u received=%u playback_ms=%u\n",snapshot.sent_packets,
               snapshot.received_packets,snapshot.playback_ms);
        printf("Voice level gain=%u codec_volume=%d peak_in=%u peak_out=%u\n",snapshot.playback_gain,
            product_codec_volume_get(),snapshot.source_peak,snapshot.output_peak);
        printf("Voice input mode=%s local_muted=%d\n",panel_voice_input_mode_name(snapshot.input_mode),snapshot.microphone_muted);
        printf("Voice detail: %s\n",snapshot.message);
        return 0;
    }
}

static int init_console(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = "panel>";
    repl_config.task_stack_size = 10 * 1024;
    repl_config.task_priority = 5;
    repl_config.max_cmdline_length = 1024;

#if CONFIG_ESP_CONSOLE_UART
    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_config, &repl_config, &repl));
#elif CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    esp_console_dev_usb_serial_jtag_config_t usbjtag_config = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&usbjtag_config, &repl_config, &repl));
#endif

    esp_console_cmd_t cmds[] = {
        { .command = "stop",          .help = "Stop camera preview\n",               .func = stop_cli },
        { .command = "i",             .help = "System status\r\n",           .func = sys_cli },
        { .command = "wifi",          .help = "wifi ssid psw\r\n",           .func = wifi_cli },
        { .command = "wifi-reconnect", .help = "Reconnect saved Wi-Fi station", .func = wifi_reconnect_cli },
        { .command = "nyx-key",       .help = "nyx-key auth_token\r\n",      .func = nyx_key_cli },
        { .command = "nyx-secret", .help = "Save Nyx signing secret", .func = nyx_secret_cli },
        { .command = "nyx-node",      .help = "nyx-node node_id\r\n",        .func = nyx_node_cli },
        { .command = "nyx-slug",      .help = "nyx-slug service_slug\r\n",   .func = nyx_slug_cli },
        { .command = "nyx-status",    .help = "Print reprovisioning status", .func = nyx_status_cli },
        { .command = "screen",        .help = "Open a native screen", .func = screen_cli },
        { .command = "nyx-login",     .help = "Panel Agent Key login start/status/cancel; no secret tokens", .func = nyx_login_cli },
        { .command = "nyx-voice",     .help = "Voice status or local SDP test; no provider call", .func = nyx_voice_cli },
        { .command = "provision-mode", .help = "Enter provisioning mode",     .func = provision_mode_cli },
        { .command = "exit-provision", .help = "Exit provisioning mode",      .func = exit_provision_cli },
        { .command = "factory-reset", .help = "Wipe all NVS config",         .func = factory_reset_cli },
        { .command = "reboot",        .help = "Soft reboot",                 .func = reboot_cli },
        { .command = "loglevel",      .help = "loglevel <tag> <D|I|W|E|N>",  .func = loglevel_cli },
    };
    for (int i = 0; i < sizeof(cmds) / sizeof(cmds[0]); i++) {
        ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));
    }
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
    return 0;
}

// --- Network event handling ---

static atomic_bool wifi_fallback_tried;
static atomic_bool fallback_requested;

static void on_net_manager_event(const net_manager_snapshot_t *snapshot, void *user)
{
    (void)user;
    lcd_renderer_set_net(snapshot);
    nyx_node_on_network(snapshot->wifi_ip || snapshot->eth_ip);
}

static int network_event_handler(bool connected)
{
    net_manager_note_legacy_network(connected);
    if (connected) {
        atomic_store(&wifi_fallback_tried, false);
        provisioning_mode_note_network_up();
        ESP_LOGI(TAG, "Network connected");
    } else {
        if (provisioning_mode_active()) {
            return 0;
        }
        // Driver stop/start must run outside its event callback.
        atomic_store(&fallback_requested, true);
    }
    return 0;
}


static atomic_bool ready;
static atomic_bool want_preview;
static atomic_bool preview_active;
static atomic_int preview_error;

bool panel_runtime_ready(void) { return atomic_load(&ready); }
void panel_runtime_set_preview(bool enabled) { atomic_store(&want_preview, enabled); }
bool panel_runtime_preview_ready(void) { return atomic_load(&preview_active); }
esp_err_t panel_runtime_preview_error(void) { return atomic_load(&preview_error); }

static void service_task(void *arg)
{
    // Art preparation runs alongside hardware initialization. Show the actual
    // Panel page immediately; no splash timer or network prerequisite.
    ESP_ERROR_CHECK(lcd_renderer_start());
    lcd_renderer_reload_config();
    bsp_display_backlight_on();
    ESP_LOGI(TAG, "Nyx Panel visible");
    setenv("TZ", "CST-8", 1);
    tzset();
    media_mode_init();
    qr_scanner_start();
    init_console();
    (void)devices_find_self();
    (void)nyx_creds_load();
    panel_voice_init();
    size_t count = 0;
    const nyx_route_t *routes = local_api_routes(&count);
    nyx_node_init();
    nyx_node_set_routes(routes, count);
    nyx_node_set_route_wrapper(local_api_route_with_logging);
    local_api_init();
    // Hosted transport initialization may wait for the C6. The live Panel UI
    // and recovery console are already running before that hardware dependency.
    ESP_ERROR_CHECK(network_init(NULL,NULL,NULL));
    net_manager_subscribe(on_net_manager_event, NULL);
    net_manager_init();
    (void)network_init(NULL, NULL, network_event_handler);
    start_clock_if_connected();
    esp_err_t http_err = local_http_server_start();
    nyx_node_start();
    // Registration is optional; an unprovisioned panel still serves recovery APIs.
    net_manager_snapshot_t snapshot = net_manager_snapshot();
    on_net_manager_event(&snapshot, NULL);
    atomic_store(&ready, http_err == ESP_OK);
    bool applied_preview = false;
    bool refreshed_bots = false;
    while (1) {
        start_clock_if_connected();
        // Refresh display metadata in the background after the live UI is up.
        if (!refreshed_bots && network_is_connected() && nyx_creds_have_agent())
            refreshed_bots = panel_voice_refresh_agents() == ESP_OK;
        if (atomic_exchange(&fallback_requested, false) && !provisioning_mode_active() &&
            !atomic_load(&wifi_fallback_tried) && !network_is_connected()) {
            atomic_store(&wifi_fallback_tried, true);
            cfg_wifi_t wf = {0};
            nvs_config_get_wifi(&wf);
            if (wf.ssid_2[0]) network_connect_wifi(wf.ssid_2, wf.psw_2);
        }
        bool wanted = atomic_load(&want_preview);
        if (wanted != applied_preview) {
            atomic_store(&preview_active, false);
            esp_err_t err = wanted ? media_mode_request_preview("native Camera") :
                                     media_mode_request_idle("native Camera closed");
            atomic_store(&preview_error, err);
            if (err == ESP_OK) {
                applied_preview = wanted;
                atomic_store(&preview_active, wanted);
            }
        }
        qr_scanner_poll();
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

esp_err_t panel_runtime_start(void)
{
    static bool started;
    if (started) return ESP_OK;
    if (xTaskCreate(service_task, "panel_services", 12288, NULL, 5, NULL) != pdPASS)
        return ESP_ERR_NO_MEM;
    started = true;
    return ESP_OK;
}
