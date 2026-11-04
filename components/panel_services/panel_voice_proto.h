#pragma once
#include "cJSON.h"
#include <stdbool.h>
#include <stddef.h>

#define PANEL_VOICE_MAX_BOTS 32
typedef struct {
    char agent_id[40], conversation_id[80], name[164];
} panel_voice_bot_t;
typedef enum {
    PANEL_VOICE_INPUT_HOLD = 0,
    PANEL_VOICE_INPUT_AUTOMATIC = 1,
} panel_voice_input_mode_t;

const char *panel_voice_input_mode_name(panel_voice_input_mode_t mode);
// All authority/readiness gates apply equally to both input modes.
bool panel_voice_capture_allowed(panel_voice_input_mode_t mode, bool held,
    bool local_muted, bool controls, bool connected, bool ready, bool server_muted);

bool panel_voice_uuid(const char *value);
bool panel_voice_conversation(const char *value);
bool panel_voice_parse_agents(const cJSON *root, panel_voice_bot_t *bots, size_t *count);
// Uses the account's explicit selection; never silently switches provider or billing source.
cJSON *panel_voice_preferences(const cJSON *discovery, const cJSON *options,
    panel_voice_input_mode_t mode);
typedef struct { char user[513], bot[513]; } panel_voice_captions_t;
// Keep server speaker identity separate from presentation; truncate at UTF-8 boundaries.
bool panel_voice_parse_captions(const cJSON *snapshot, panel_voice_captions_t *out);

typedef struct { char request_id[40], text[301]; bool microphone_muted; } panel_voice_remote_t;
bool panel_voice_parse_remote(const void *json, size_t size, bool open, panel_voice_remote_t *out);
// Returns fixed local text for recognized server errors; never reflects remote prose.
const char *panel_voice_error_message(const cJSON *root, int http_status);

// Caller frees the normalized, bounded initial WebRTC offer.
char *panel_voice_prepare_offer(const char *sdp, size_t size);

// Accept the bounded provider readiness event, which may include a large session.
bool panel_voice_provider_started(const char *json, size_t size);

// Normal server termination must not become a transport error on socket close.
bool panel_voice_normal_end(const cJSON *session);
