// Opt-in original C161/B40/history research adapter. The application retains
// its existing single I2S -> AEC -> AGC -> FIFO producer and native PCM speed.
#include "panel_wake_runtime.h"
#include "portable/c161_binary40_history_events_v1.h"
#include "parallel_model.h"
#include <algorithm>
#include <cmath>
#include <new>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern "C" {
#include "frontend.h"
#include "frontend_util.h"
}

extern const uint8_t c161_bytes[] asm("_binary_c161_bin_start");
extern const uint8_t binary40_bytes[] asm("_binary_binary40_bin_start");
extern const uint8_t history_bytes[] asm("_binary_history_expert_bin_start");

namespace {
constexpr int chunk_samples = 480;
using panel_wake_parallel::Model;
using panel_wake_parallel::tag;


struct ParallelDetector {
    Model models[3];
    FrontendState frontend{};
    nyx::TripleEvents events;
    float features[120]{};
    unsigned feature_count = 0;
    bool frontend_started = false, failed = false;
    uint64_t steps = 0, inference_us = 0;
    unsigned inference_max_us = 0;

    ~ParallelDetector() { if (frontend_started) FrontendFreeStateContents(&frontend); }

    bool initialize() {
        if (!models[0].initialize(c161_bytes, "c161", false) ||
            !models[1].initialize(binary40_bytes, "binary40", true) ||
            !models[2].initialize(history_bytes, "history_expert", true)) return false;
        FrontendConfig config{};
        FrontendFillConfigWithDefaults(&config);
        config.window.size_ms = 30; config.window.step_size_ms = 10;
        config.filterbank.num_channels = 40;
        config.filterbank.lower_band_limit = 125; config.filterbank.upper_band_limit = 7500;
        config.noise_reduction.smoothing_bits = 10;
        config.noise_reduction.even_smoothing = .025; config.noise_reduction.odd_smoothing = .06;
        config.noise_reduction.min_signal_remaining = .05;
        config.pcan_gain_control.enable_pcan = 1; config.pcan_gain_control.strength = .95;
        config.pcan_gain_control.offset = 80; config.pcan_gain_control.gain_bits = 21;
        config.log_scale.enable_log = 1; config.log_scale.scale_shift = 6;
        frontend_started = true;
        return FrontendPopulateState(&config, &frontend, 16000);
    }

    int detect(const int16_t* pcm) {
        if (failed || !pcm) return -1;
        bool triggered = false;
        for (size_t at = 0; at < chunk_samples;) {
            size_t consumed = 0;
            auto frame = FrontendProcessSamples(&frontend, pcm + at,
                std::min(size_t(160), size_t(chunk_samples) - at), &consumed);
            if (!consumed || consumed > size_t(chunk_samples) - at ||
                (frame.size && (frame.size != 40 || feature_count >= 3))) { failed = true; return -1; }
            at += consumed;
            if (!frame.size) continue;
            for (unsigned i = 0; i < 40; ++i)
                features[feature_count * 40 + i] = std::clamp(frame.values[i] / 25.6f, 0.f, 26.f);
            if (++feature_count != 3) continue;
            feature_count = 0;
            uint8_t codes[3]{};
            int64_t started = esp_timer_get_time();
            for (int i = 0; i < 3; ++i) {
                if (!models[i].invoke(features, codes[i])) {
                    ESP_LOGE(tag, "Inference/cache failure model=%s; stopping all detectors", models[i].name);
                    failed = true; return -1;
                }
            }
            unsigned elapsed = unsigned(esp_timer_get_time() - started);
            inference_us += elapsed; inference_max_us = std::max(inference_max_us, elapsed);
            ++steps;
            auto event = events.update(codes[0], codes[1], codes[2]);
            if (event.emitted) {
                triggered = true;
                ESP_LOGI(tag, "activation step=%llu emitted=%u detected=%u suppressed=%u",
                         (unsigned long long)steps, unsigned(event.emitted), unsigned(event.detected), unsigned(event.suppressed));
            }
            if (steps % 167 == 0) ESP_LOGI(tag, "progress step=%llu mean_us=%u max_us=%u stack_free=%u",
                (unsigned long long)steps, unsigned(inference_us / steps), inference_max_us,
                unsigned(uxTaskGetStackHighWaterMark(nullptr)));
        }
        return triggered ? 1 : 0;
    }
};

model_iface_data_t* create(const void*, det_mode_t) {
    auto* detector = new(std::nothrow)ParallelDetector;
    if (!detector) return nullptr;
    if (!detector->initialize()) { ESP_LOGE(tag, "Parallel initialization failed"); delete detector; return nullptr; }
    return reinterpret_cast<model_iface_data_t*>(detector);
}
void destroy(model_iface_data_t* data) { delete reinterpret_cast<ParallelDetector*>(data); }
wakenet_state_t detect(model_iface_data_t* data, int16_t* pcm) {
    return static_cast<wakenet_state_t>(reinterpret_cast<ParallelDetector*>(data)->detect(pcm));
}
int chunk(model_iface_data_t*) { return chunk_samples; }
int rate(model_iface_data_t*) { return 16000; }
int channels(model_iface_data_t*) { return 1; }
} // namespace

extern "C" const esp_wn_iface_t panel_wake_runtime_iface = {
    .create=create, .get_start_point=nullptr, .get_samp_chunksize=chunk,
    .get_channel_num=channels, .get_samp_rate=rate, .get_word_num=nullptr,
    .get_word_name=nullptr, .set_det_threshold=nullptr, .reset_det_threshold=nullptr,
    .get_det_threshold=nullptr, .get_triggered_channel=nullptr, .get_vol_gain=nullptr,
    .detect=detect, .detect_mfcc=nullptr, .get_mfcc_data=nullptr, .clean=nullptr, .destroy=destroy
};
extern "C" int panel_wake_runtime_prime_samples(void) { return 81600; }
