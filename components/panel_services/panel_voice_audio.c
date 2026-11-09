#include "panel_voice_audio.h"
#include <string.h>
#include <math.h>

bool panel_voice_pcm_push(panel_voice_pcm_t *fifo, const int16_t *pcm, size_t samples)
{
    if (!fifo || !fifo->samples || !fifo->capacity || !pcm || samples>fifo->capacity-fifo->count) return false;
    size_t tail=(fifo->head+fifo->count)%fifo->capacity;
    size_t first=samples<fifo->capacity-tail?samples:fifo->capacity-tail;
    memcpy(fifo->samples+tail,pcm,first*sizeof(*pcm));
    memcpy(fifo->samples,pcm+first,(samples-first)*sizeof(*pcm));
    fifo->count+=samples;
    return true;
}
bool panel_voice_pcm_pop(panel_voice_pcm_t *fifo, int16_t *pcm, size_t samples)
{
    if (!fifo || !fifo->samples || !fifo->capacity || !pcm || samples>fifo->count) return false;
    size_t first=samples<fifo->capacity-fifo->head?samples:fifo->capacity-fifo->head;
    memcpy(pcm,fifo->samples+fifo->head,first*sizeof(*pcm));
    memcpy(pcm+first,fifo->samples,(samples-first)*sizeof(*pcm));
    fifo->head=(fifo->head+samples)%fifo->capacity;
    fifo->count-=samples;
    return true;
}
void panel_voice_pcm_clear(panel_voice_pcm_t *fifo)
{
    if (!fifo) return;
    if (fifo->samples) memset(fifo->samples,0,fifo->capacity*sizeof(*fifo->samples));
    fifo->head=fifo->count=0;
}
void panel_voice_pcm_preroll_push(panel_voice_pcm_t *fifo, const int16_t *pcm, size_t samples)
{
    if(!fifo || !fifo->samples || !fifo->capacity || !pcm)return;
    if(samples>fifo->capacity){pcm+=samples-fifo->capacity;samples=fifo->capacity;}
    size_t discard=samples>fifo->capacity-fifo->count?samples-(fifo->capacity-fifo->count):0;
    fifo->head=(fifo->head+discard)%fifo->capacity;fifo->count-=discard;
    (void)panel_voice_pcm_push(fifo,pcm,samples);
}

bool panel_voice_pcm_gate(panel_voice_pcm_t *fifo, bool *started, bool ready, bool muted)
{
    if (!fifo || !started) return false;
    if (ready) *started=true;
    bool admit=!muted && (ready || !*started);
    if (!admit && fifo->count) panel_voice_pcm_clear(fifo);
    return admit;
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
