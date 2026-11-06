#include "panel_voice_audio.h"
#include <string.h>
#include <math.h>

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

void panel_voice_ringback(uint32_t *position, int16_t *pcm, size_t samples)
{
    if(!position || !pcm)return;
    uint32_t tick=*position%48000;
    for(size_t i=0;i<samples;++i) {
        float envelope=0;
        if(tick<16000) {
            uint32_t edge=tick<80?tick:tick>15920?16000-tick:80;
            envelope=(float)edge/80;
        }
        if(envelope>0) {
            float t=(float)tick*(6.28318530718f/16000);
            pcm[i]=(int16_t)(envelope*1800*(sinf(t*440)+sinf(t*480)));
        }else pcm[i]=0;
        if(++tick==48000)tick=0;
    }
    *position=tick;
}
