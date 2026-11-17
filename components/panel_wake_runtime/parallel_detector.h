#pragma once
#include "parallel_model.h"
#include "portable/c161_binary40_history_events_v1.h"
#include <algorithm>
#include <cstdint>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern "C" {
#include "frontend.h"
#include "frontend_util.h"
}

namespace panel_wake_parallel {
constexpr int chunk_samples = 480;
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

    bool initialize(const uint8_t* c161, const uint8_t* binary40, const uint8_t* history) {
        if (!models[0].initialize(c161, "c161", false) ||
            !models[1].initialize(binary40, "binary40", true) ||
            !models[2].initialize(history, "history_expert", true)) return false;
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

} // namespace panel_wake_parallel
