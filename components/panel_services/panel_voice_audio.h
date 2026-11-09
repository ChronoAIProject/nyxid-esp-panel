#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Lossless PCM FIFO, serialized by its caller. A full write fails atomically:
// connection delays must never silently discard the start of an utterance.
typedef struct {
    int16_t *samples;
    size_t capacity, head, count;
} panel_voice_pcm_t;
bool panel_voice_pcm_push(panel_voice_pcm_t *fifo, const int16_t *pcm, size_t samples);
bool panel_voice_pcm_pop(panel_voice_pcm_t *fifo, int16_t *pcm, size_t samples);
void panel_voice_pcm_clear(panel_voice_pcm_t *fifo);
// Idle-only lookback: retain the newest samples before wake detection. Calls
// use the lossless push above; this policy must never overwrite call speech.
void panel_voice_pcm_preroll_push(panel_voice_pcm_t *fifo, const int16_t *pcm, size_t samples);

// Voice gain with a soft knee; bounded input never wraps or hard clips.
int16_t panel_voice_amplify_sample(int16_t sample, unsigned gain);

// 16 kHz local ringback, one second on and two seconds off, with soft edges.
// Independent of provider playback timing and its gain setting.
void panel_voice_ringback(uint32_t *position, int16_t *pcm, size_t samples);
