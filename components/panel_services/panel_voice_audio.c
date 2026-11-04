#include "panel_voice_audio.h"
#include <string.h>

void panel_voice_reference_push(panel_voice_reference_t *ref, const int16_t *pcm, size_t samples)
{
    if (!ref || !pcm) return;
    if (samples >= PANEL_VOICE_REFERENCE_SAMPLES) {
        pcm += samples - PANEL_VOICE_REFERENCE_SAMPLES;
        samples = PANEL_VOICE_REFERENCE_SAMPLES;
        ref->head = ref->count = 0;
    }
    for (size_t i=0; i<samples; ++i) {
        if (ref->count == PANEL_VOICE_REFERENCE_SAMPLES) {
            ref->head = (ref->head + 1) % PANEL_VOICE_REFERENCE_SAMPLES;
            --ref->count;
        }
        ref->samples[(ref->head + ref->count++) % PANEL_VOICE_REFERENCE_SAMPLES] = pcm[i];
    }
}

void panel_voice_reference_read(panel_voice_reference_t *ref, int16_t *pcm, size_t samples)
{
    if (!pcm) return;
    memset(pcm, 0, samples * sizeof(*pcm));
    if (!ref) return;
    for (size_t i=0; i<samples && ref->count; ++i) {
        pcm[i] = ref->samples[ref->head];
        ref->head = (ref->head + 1) % PANEL_VOICE_REFERENCE_SAMPLES;
        --ref->count;
    }
}

int16_t panel_voice_amplify_sample(int16_t sample, unsigned gain)
{
    if (gain>8) gain=8;
    int32_t value=(int32_t)sample*(int32_t)gain;
    int32_t magnitude=value<0 ? -value : value;
    const int32_t knee=24576, headroom=32767-knee;
    if (magnitude>knee) {
        int32_t excess=magnitude-knee;
        magnitude=knee+(excess*headroom)/(excess+headroom);
    }
    return (int16_t)(value<0 ? -magnitude : magnitude);
}
