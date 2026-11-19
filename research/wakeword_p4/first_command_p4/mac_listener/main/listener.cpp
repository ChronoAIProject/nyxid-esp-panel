#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "c161_coded_cue_v5.h"
#include "esp_aec.h"
#include "esp_agc.h"
#include "esp_flash.h"
#include "esp_heap_caps_init.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "panel_wake_runtime.h"
extern "C" {
#include "panel_voice_audio.h"
}

namespace {
constexpr int kHop = 160, kLookback = 32000, kBlock = 480;
constexpr uint32_t kOffset = 2U<<20;
struct DspSection {
    UBaseType_t old = portSET_INTERRUPT_MASK_FROM_ISR();
    DspSection() { asm volatile("fmv.w.x ft0, zero" ::: "ft0"); }
    ~DspSection() { portCLEAR_INTERRUPT_MASK_FROM_ISR(old); }
};

int16_t* buffer(size_t n) {
    auto* ptr = static_cast<int16_t*>(heap_caps_calloc(n,sizeof(int16_t),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
    if (!ptr) abort();
    return ptr;
}

void dump(const int16_t* samples,int count) {
    static constexpr char hex[]="0123456789abcdef";
    char line[4*kHop+1];
    const auto* bytes=reinterpret_cast<const uint8_t*>(samples);
    for (int start=0; start<count; start+=kHop) {
        int n=std::min(count-start,kHop);
        for (int i=0;i<2*n;++i) {
            uint8_t b=bytes[2*start+i];
            line[2*i]=hex[b>>4]; line[2*i+1]=hex[b&15];
        }
        line[4*n]=0;
        std::printf("AUDIO %d %s\n",start,line);
    }
}

void worker(void*) {
    uint32_t h[3];
    ESP_ERROR_CHECK(esp_flash_read(nullptr,h,kOffset,sizeof(h)));
    if (h[0]!=0x584e594d || !h[2] || h[2]>400000) abort();
    const int length=h[2];
    auto* pcm=buffer(length+2048);
    ESP_ERROR_CHECK(esp_flash_read(nullptr,pcm,kOffset+sizeof(h),length*2));
    const auto& wn=panel_wake_runtime_iface;
    model_iface_data_t* model;
    aec_handle_t* aec;
    void* agc;
    {
        DspSection lock;
        model=wn.create(nullptr,DET_MODE_95);
        aec_config_t cfg{};
        cfg.mic_num=cfg.ref_num=cfg.out_num=1; cfg.filter_length=8; cfg.sample_rate=16000;
        cfg.caps=MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT; cfg.mode=AEC_MODE_VOIP_HIGH_PERF; cfg.nlp_level=AEC_NLP_LEVEL_AGGR;
        aec=aec_create_from_config(&cfg);
        agc=esp_agc_open(AGC_MODE_2,16000);
        if (!model || !aec || !agc) abort();
        set_agc_config(agc,15,1,3);
    }
    const int chunk=aec_get_chunksize(aec);
    if (chunk<=0 || chunk>1024 || wn.get_samp_chunksize(model)!=kBlock) abort();
    alignas(16) int16_t wake[kBlock]{}, ref[1024]{}, clean[1024], frame[kHop], adjusted[kHop];
    int prime=0;
    while (prime<panel_wake_runtime_prime_samples()) {
        DspSection lock;
        if (wn.detect(model,wake)<0) abort();
        prime+=kBlock;
    }
    if (prime!=81600) abort();
    panel_voice_pcm_t history{buffer(kLookback),kLookback,0,0};
    panel_voice_pcm_t call{buffer(60*16000),60*16000,0,0};
    auto* processed=buffer(length+2048);
    nyx::CodedCueV5 cue;
    int fill=0, wake_fill=0, emitted=0, first=-1, peak_us=0, overruns=0, blocks=0;
    int64_t sum_us=0;
    std::printf("READY %lu %d %d %d\n",(unsigned long)h[1],length,chunk,prime);
    for (int at=0;at<length;at+=chunk) {
        int hit161=-1, hitcue=-1;
        int64_t begin=esp_timer_get_time();
        {
            DspSection lock;
            aec_process(aec,pcm+at,ref,clean);
        }
        for (int i=0;i<chunk;++i) {
            frame[fill++]=clean[i];
            if (fill!=kHop) continue;
            fill=0;
            {
                DspSection lock;
                if (esp_agc_process(agc,frame,adjusted,kHop,16000)!=ESP_AGC_SUCCESS) abort();
            }
            std::memcpy(processed+emitted,adjusted,sizeof(adjusted));
            if (first<0) panel_voice_pcm_preroll_push(&history,adjusted,kHop);
            else if (!panel_voice_pcm_push(&call,adjusted,kHop)) abort();
            emitted+=kHop;
        }
        for (int i=0;i<chunk;++i) {
            if (cue.update(pcm[at+i])) hitcue=at+i+1;
            wake[wake_fill++]=pcm[at+i];
            if (wake_fill!=kBlock) continue;
            wake_fill=0;
            DspSection lock;
            const int hit=wn.detect(model,wake);
            if (hit<0) abort();
            if (hit>0) hit161=at+i+1;
        }
        if (first<0 && (hitcue>=0 || hit161>=0)) {
            const size_t part=std::min(history.count,history.capacity-history.head);
            first=emitted-static_cast<int>(history.count);
            if (!panel_voice_pcm_push(&call,history.samples+history.head,part) ||
                !panel_voice_pcm_push(&call,history.samples,history.count-part)) abort();
            panel_voice_pcm_clear(&history);
            std::printf("OPEN %d %d %d\n",at+chunk,emitted,first);
        }
        int us=esp_timer_get_time()-begin;
        ++blocks; sum_us+=us; peak_us=std::max(peak_us,us); overruns+=us*16>chunk*1000;
        if (hit161>=0 || hitcue>=0) std::printf("HITS %d %d %d\n",at+chunk,hit161,hitcue);
        if (blocks%64==0) vTaskDelay(1);
    }
    if (first>=0 && (call.count!=static_cast<size_t>(emitted-first) ||
        std::memcmp(call.samples,processed+first,call.count*sizeof(int16_t)))) abort();
    std::printf("FINISH %d %d %u %d %lld %d %d %u\n",emitted,first,(unsigned)call.count,blocks,
                (long long)sum_us,peak_us,overruns,(unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
    if (first>=0) dump(call.samples,call.count);
    wn.destroy(model); aec_destroy(aec); esp_agc_close(agc);
    std::puts("LISTENER PASS");
    vTaskDelete(nullptr);
}
}
extern "C" void app_main() {
    const uint32_t caps[]={MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT|MALLOC_CAP_32BIT,0,0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps,0x48000000,0x49000000));
    if (xTaskCreate(worker,"listener",48*1024,nullptr,3,nullptr)!=pdPASS) abort();
}
