#pragma once
#include <stddef.h>
#include <stdint.h>

// Bounded playback reference for the 16 kHz acoustic echo canceller. The caller
// serializes push/read; old samples are discarded when playback gets ahead.
#define PANEL_VOICE_REFERENCE_SAMPLES 2048
typedef struct {
    int16_t samples[PANEL_VOICE_REFERENCE_SAMPLES];
    size_t head, count;
} panel_voice_reference_t;
void panel_voice_reference_push(panel_voice_reference_t *ref, const int16_t *pcm, size_t samples);
void panel_voice_reference_read(panel_voice_reference_t *ref, int16_t *pcm, size_t samples);

// Voice gain with a soft knee; bounded input never wraps or hard clips.
int16_t panel_voice_amplify_sample(int16_t sample, unsigned gain);
