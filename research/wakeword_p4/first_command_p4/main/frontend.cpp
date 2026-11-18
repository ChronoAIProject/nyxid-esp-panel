#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "c161_coded_cue_v5.h"
#include "esp_aec.h"
#include "esp_agc.h"
#include "esp_heap_caps.h"
#include "esp_heap_caps_init.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {
extern const uint8_t digital[] asm("_binary_4090_c161_coded_cue_v5_command_digital_wav_start");
extern const uint8_t near_wave[] asm("_binary_4090_c161_coded_cue_v5_command_near_wav_start");
extern const uint8_t far_wave[] asm("_binary_4090_c161_coded_cue_v5_command_far_wav_start");

constexpr int rate = 16000, frame_samples = 160, max_chunk = 1024;

// A tick can suspend floating-point state immediately before a rounding-mode CSR.
// QEMU misses the lazy-FPU trap for that CSR. Mask ticks only during DSP calls.
struct DspSection {
    UBaseType_t previous_mask = portSET_INTERRUPT_MASK_FROM_ISR();
    DspSection() { asm volatile("fmv.w.x ft0, zero" ::: "ft0"); }
    ~DspSection() { portCLEAR_INTERRUPT_MASK_FROM_ISR(previous_mask); }
};

void emit(const char* stage, int at, const int16_t* pcm) {
    static constexpr char hex[] = "0123456789abcdef";
    char line[4 * frame_samples + 1];
    for (int i = 0; i < frame_samples; ++i) {
        uint16_t value = static_cast<uint16_t>(pcm[i]);
        line[4*i]   = hex[(value >> 4) & 15];
        line[4*i+1] = hex[value & 15];
        line[4*i+2] = hex[value >> 12];
        line[4*i+3] = hex[(value >> 8) & 15];
    }
    line[sizeof(line)-1] = 0;
    std::printf("PCM %s %d %s\n", stage, at, line);
}

void run(const char* channel, const uint8_t* wav) {
    aec_config_t config{};
    config.mic_num = config.ref_num = config.out_num = 1;
    config.filter_length = 8;
    config.sample_rate = rate;
    config.caps = MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM;
    config.mode = AEC_MODE_VOIP_HIGH_PERF;
    config.nlp_level = AEC_NLP_LEVEL_AGGR;
    aec_handle_t* aec;
    void* agc;
    {
        DspSection dsp;
        aec = aec_create_from_config(&config);
        agc = esp_agc_open(AGC_MODE_2, rate);
        if (!aec || !agc) abort();
        set_agc_config(agc, 15, 1, 3);
    }
    const int chunk = aec_get_chunksize(aec);
    uint32_t bytes;
    std::memcpy(&bytes, wav + 40, 4);
    if (std::memcmp(wav + 36, "data", 4) || !bytes || bytes % 2 || chunk <= 0 || chunk > max_chunk) abort();
    const int length = bytes / 2;
    const int prime = ((81600 + chunk - 1) / chunk) * chunk;
    std::printf("BEGIN %s %d %d %d\n", channel, length, chunk, prime);
    alignas(16) int16_t mic[max_chunk], ref[max_chunk]{}, clean[max_chunk];
    alignas(16) int16_t frame[frame_samples], adjusted[frame_samples];
    nyx::CodedCueV5 detector;
    int fill = 0, emitted = -prime;
    for (int at = -prime; at < length + 2048; at += chunk) {
        if ((at + prime) % (64 * chunk) == 0) vTaskDelay(1);
        std::memset(mic, 0, chunk * sizeof(*mic));
        if (at >= 0 && at < length) {
            std::memcpy(mic, wav + 44 + at*2, std::min(chunk, length-at) * sizeof(*mic));
        }
        {
            DspSection dsp;
            aec_process(aec, mic, ref, clean);
        }
        for (int i = 0; i < chunk; ++i) {
            frame[fill++] = clean[i];
            if (fill < frame_samples) continue;
            fill = 0;
            if (emitted + frame_samples > 0) emit("aec", emitted, frame);
            {
                DspSection dsp;
                if (esp_agc_process(agc, frame, adjusted, frame_samples, rate) != ESP_AGC_SUCCESS) abort();
            }
            for (int j = 0; j < frame_samples; ++j) {
                if (detector.update(adjusted[j])) std::printf("EVENT %s %d\n", channel, emitted+j+1);
            }
            if (emitted + frame_samples > 0) emit("agc", emitted, adjusted);
            emitted += frame_samples;
        }
    }
    esp_agc_close(agc);
    aec_destroy(aec);
    std::printf("END %s %d\n", channel, emitted);
}

void worker(void*) {
    run("digital", digital);
    run("near", near_wave);
    run("far", far_wave);
    std::puts("FRONTEND PASS");
    vTaskDelete(nullptr);
}
}  // namespace

extern "C" void app_main() {
    // QEMU maps ordinary RAM here without emulating P4 PSRAM initialization.
    const uint32_t caps[] = {MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_32BIT, 0, 0};
    ESP_ERROR_CHECK(heap_caps_add_region_with_caps(caps, 0x48000000, 0x49000000));
    if (xTaskCreate(worker, "voice_processing", 48*1024, nullptr, 3, nullptr) != pdPASS) abort();
}
