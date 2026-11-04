#pragma once
#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include "panel_voice_proto.h"

typedef struct {
    bool configured, busy, active, holding;
    unsigned sent_packets, received_packets, playback_ms;
    char state[24], message[128], service_id[40], agent_id[40], agent_name[164];
    unsigned bots_revision, playback_gain, source_peak, output_peak;
    panel_voice_input_mode_t input_mode;
    panel_voice_input_mode_t preferred_input_mode;
    bool microphone_muted;
    char session_id[48], remote_request_id[40], remote_state[24];
    bool remote_initiated;
} panel_voice_snapshot_t;

void panel_voice_init(void);
void panel_voice_snapshot(panel_voice_snapshot_t *out);
esp_err_t panel_voice_configure(const char *agent_id);
esp_err_t panel_voice_refresh_agents(void);
size_t panel_voice_bots(panel_voice_bot_t *out, size_t cap);
esp_err_t panel_voice_start(void);
void panel_voice_stop(void);
void panel_voice_hold(bool pressed);
esp_err_t panel_voice_gain_set(unsigned gain);
esp_err_t panel_voice_volume_set(unsigned volume);
// Mode changes require an idle call and persist in this server's profile.
esp_err_t panel_voice_input_mode_set(panel_voice_input_mode_t mode);
void panel_voice_microphone_mute(bool muted);
esp_err_t panel_voice_remote_open(const panel_voice_remote_t *request, bool *duplicate);
esp_err_t panel_voice_remote_say(const panel_voice_remote_t *request, bool *duplicate);
// Bounded codec/mic/speaker exercise without a provider or an account session.
esp_err_t panel_voice_test(char *json, size_t cap);
esp_err_t panel_voice_peer_test(char *json, size_t cap);
