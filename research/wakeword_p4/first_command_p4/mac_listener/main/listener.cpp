#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "c161_coded_cue_reference_v5.h"
#ifdef NYX_REPEAT_REFERENCE
#include "repeated_cue_reference.h"
#endif
#include "reference_capture.h"
#include "reference_activity.h"
#include "reference_prefix.h"
#include "reference_signature.h"
#include "reference_wake.h"
#ifdef NYX_SPEEX_CAPTURE
#include "speex/speex_echo.h"
#endif
#ifdef NYX_CAPTURE_OPUS
#include "capture_opus.h"
#endif
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

#ifndef NYX_CAPTURE_OPUS
void dump(const char* label, const int16_t* samples,int count) {
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
        std::printf("%s %d %s\n",label,start,line);
    }
}

#endif

void worker(void*) {
    uint32_t h[3];
    ESP_ERROR_CHECK(esp_flash_read(nullptr,h,kOffset,sizeof(h)));
    bool call_setup = false;
#ifdef NYX_CAPTURE_OPUS
    int setup_us = 0;
    call_setup = h[0] == 0x4f52594e || h[0] == 0x5752594e;
    if (call_setup) {
        vTaskPrioritySet(nullptr,4);
        setup_us = h[0] == 0x5752594e ? 1500000 : 0;
    }
#endif
    const bool qualify_capture_frame = h[0] == 0x4652594e;
    const bool frame_activity = qualify_capture_frame || h[0] == 0x4152594e;
    const bool wait_for_third = frame_activity || h[0] == 0x5452594e;
    const bool signed_reference = call_setup || wait_for_third || h[0] == 0x4452594e;
    const bool prefix_guarded = signed_reference || h[0] == 0x5052594e;
    const bool guarded = prefix_guarded || h[0] == 0x4752594e;
    const bool stereo = guarded || h[0] == 0x5352594e;
    if ((!stereo && h[0]!=0x584e594d) || !h[2] || h[2]>400000) abort();
    const int length=h[2];
    auto* pcm=buffer(length+2048);
    ESP_ERROR_CHECK(esp_flash_read(nullptr,pcm,kOffset+sizeof(h),length*2));
    auto* playback = stereo ? buffer(length+2048) : nullptr;
    if (stereo) ESP_ERROR_CHECK(esp_flash_read(nullptr,playback,kOffset+sizeof(h)+length*2,length*2));
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
#ifdef NYX_SPEEX_CAPTURE
    SpeexEchoState* speex = nullptr;
    if (stereo) {
        const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
        size_t before = heap_caps_get_free_size(caps);
        int rate = 16000;
        {
            DspSection lock;
            speex = speex_echo_state_init(kHop, 2048);
            if (!speex || speex_echo_ctl(speex, SPEEX_ECHO_SET_SAMPLING_RATE, &rate)) abort();
        }
        std::printf("SPEEX_HEAP %u %u\n", unsigned(before), unsigned(heap_caps_get_free_size(caps)));
    }
#endif
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
    auto* selected = stereo ? buffer(length+2048) : processed;
    nyx::ReferenceWake model_route, cue_route;
    nyx::ReferenceCapture capture;
    nyx::ReferenceActivity capture_activity, wake_activity;
#ifdef NYX_REPEAT_REFERENCE
    nyx::RepeatedCueReference guard;
#else
    nyx::CodedCueReferenceV5 guard;
#endif
    nyx::ReferencePrefix prefix;
    nyx::ReferenceSignature signature;
    nyx::CodedCueV5 cue;
    int fill=0, wake_fill=0, emitted=0, first=-1, peak_us=0, overruns=0, blocks=0;
    int64_t sum_us=0;
    std::printf("READY %lu %d %d %d\n",(unsigned long)h[1],length,chunk,prime);
#ifdef NYX_CAPTURE_OPUS
    CaptureOpus codec(call, length+chunk, chunk*1000/16, setup_us);
#endif
    for (int at=0;at<length;at+=chunk) {
#ifdef NYX_CAPTURE_OPUS
        codec.next_block();
#endif
        int hit161=-1, hitcue=-1, allowed_cue=-1, allowed161=-1;
        int64_t begin=esp_timer_get_time();
        {
            DspSection lock;
            aec_process(aec,pcm+at,stereo ? playback+at : ref,clean);
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
            if (stereo) {
                // AGC has already completed this frame. Its reference is available
                // through the same last sample before the FIFO is written.
                static_assert(kHop == nyx::ReferenceActivity::kFrame);
                const bool loud_frame = !qualify_capture_frame || nyx::ReferenceActivity::active_frame(playback+emitted);
                for (int j=0;j<kHop;++j) {
                    const int t=emitted+j;
                    const bool active = frame_activity && !qualify_capture_frame ? capture_activity.update(playback[t])
                        : loud_frame && std::abs(int(playback[t])) >= 64;
                    selected[t]=capture.update_active(pcm[t],active,adjusted[j]);
                }
            }
#ifdef NYX_SPEEX_CAPTURE
            if (speex) {
                DspSection lock;
                speex_echo_cancellation(speex, pcm+emitted, playback+emitted, selected+emitted);
            }
#endif
            if (first<0) panel_voice_pcm_preroll_push(&history,selected+emitted,kHop);
#ifdef NYX_CAPTURE_OPUS
            else codec.push(selected+emitted,kHop);
#else
            else if (!panel_voice_pcm_push(&call,selected+emitted,kHop)) abort();
#endif
            emitted+=kHop;
        }
        for (int i=0;i<chunk;++i) {
            const bool matched=cue.update(pcm[at+i]);
            if (matched) hitcue=at+i+1;
            const int16_t reference = stereo ? playback[at+i] : 0;
            const bool active = frame_activity ? wake_activity.update(reference) : std::abs(int(reference)) >= 64;
            const bool full_clear = !stereo || guard.allow(reference,true);
            if (signed_reference) signature.update(reference);
            using PrefixState = nyx::ReferencePrefix::State;
            const auto prefix_state = prefix_guarded ? prefix.update(reference, [&] {
                return !signed_reference || signature.matches();
            }) : PrefixState::clear;
            const bool reference_clear = full_clear && prefix_state != PrefixState::confirmed;
            const bool awaiting_third = wait_for_third && prefix_state == PrefixState::awaiting_third;
            const bool cue_ok = guarded ? cue_route.update_active(active,reference_clear,matched,awaiting_third) : matched && reference_clear;
            if (cue_ok) allowed_cue=at+i+1;
            bool model_hit=false;
            wake[wake_fill++]=pcm[at+i];
            if (wake_fill==kBlock) {
                wake_fill=0;
                DspSection lock;
                const int hit=wn.detect(model,wake);
                if (hit<0) abort();
                model_hit=hit>0;
                if (model_hit) hit161=at+i+1;
            }
            if (guarded ? model_route.update_active(active,reference_clear,model_hit,awaiting_third) : model_hit) allowed161=at+i+1;
        }
        if (first<0 && (allowed_cue>=0 || allowed161>=0)) {
            const size_t part=std::min(history.count,history.capacity-history.head);
            first=emitted-static_cast<int>(history.count);
            if (!panel_voice_pcm_push(&call,history.samples+history.head,part) ||
                !panel_voice_pcm_push(&call,history.samples,history.count-part)) abort();
            panel_voice_pcm_clear(&history);
#ifdef NYX_CAPTURE_OPUS
            codec.open();
#endif
            std::printf("OPEN %d %d %d\n",at+chunk,emitted,first);
        }
#ifdef NYX_CAPTURE_OPUS
        codec.start_if_ready();
#endif
        int us=esp_timer_get_time()-begin;
        ++blocks; sum_us+=us; peak_us=std::max(peak_us,us); overruns+=us*16>chunk*1000;
        if (hit161>=0 || hitcue>=0) std::printf("HITS %d %d %d\n",at+chunk,hit161,hitcue);
        if (stereo && hitcue>=0) std::printf("CUE_ROUTE %d %d\n",hitcue,allowed_cue);
        if (guarded && allowed_cue>=0) std::printf("V5_ALLOW %d\n",allowed_cue);
        if (guarded && allowed161>=0) std::printf("C161_ALLOW %d\n",allowed161);
#ifndef NYX_CAPTURE_OPUS
        if (blocks%64==0) vTaskDelay(1);
#endif
    }
    const unsigned captured = first>=0 ? emitted-first : 0;
#ifdef NYX_CAPTURE_OPUS
    codec.finish(captured);
#else
    if (first>=0 && (call.count!=static_cast<size_t>(emitted-first) ||
        std::memcmp(call.samples,selected+first,call.count*sizeof(int16_t)))) abort();
#endif
    std::printf("FINISH %d %d %u %d %lld %d %d %u\n",emitted,first,captured,blocks,
                (long long)sum_us,peak_us,overruns,(unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
#ifndef NYX_CAPTURE_OPUS
    if (first>=0) {
        dump("AUDIO",call.samples,call.count);
        if (stereo) dump("PROCESSED",processed+first,call.count);
    }
#endif
#ifdef NYX_SPEEX_CAPTURE
    if (speex) speex_echo_state_destroy(speex);
#endif
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
