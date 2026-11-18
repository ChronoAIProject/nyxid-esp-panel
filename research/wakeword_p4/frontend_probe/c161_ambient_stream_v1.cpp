#include "../portable/c161_ambient_history_v1.h"
#include "../portable/c161_binary40_events_v1.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

#include <algorithm>
#include <new>

struct Stream {
    FrontendState frontend{};
    nyx::AmbientHistory ambient;
    nyx::ParallelEvents decoder;
};

extern "C" Stream* make_stream() {
    auto* stream = new (std::nothrow) Stream{};
    if (!stream) return nullptr;
    FrontendConfig config{};
    FrontendFillConfigWithDefaults(&config);
    config.window.size_ms = 30;
    config.window.step_size_ms = 10;
    config.filterbank.num_channels = 40;
    config.filterbank.lower_band_limit = 125.0f;
    config.filterbank.upper_band_limit = 7500.0f;
    config.noise_reduction.smoothing_bits = 10;
    config.noise_reduction.even_smoothing = .025f;
    config.noise_reduction.odd_smoothing = .06f;
    config.noise_reduction.min_signal_remaining = .05f;
    config.pcan_gain_control.enable_pcan = 1;
    config.pcan_gain_control.strength = .95f;
    config.pcan_gain_control.offset = 80.0f;
    config.pcan_gain_control.gain_bits = 21;
    config.log_scale.enable_log = 1;
    config.log_scale.scale_shift = 6;
    if (!FrontendPopulateState(&config, &stream->frontend, 16000)) {
        FrontendFreeStateContents(&stream->frontend);
        delete stream;
        return nullptr;
    }
    return stream;
}

extern "C" void destroy_stream(Stream* stream) {
    FrontendFreeStateContents(&stream->frontend);
    delete stream;
}

extern "C" int stream_hop(Stream* stream, const int16_t* pcm, int speaking, uint16_t* out) {
    auto& state = stream->frontend;
    stream->ambient.update(speaking != 0, state.noise_reduction.estimate);
    size_t consumed = 0;
    auto result = FrontendProcessSamples(&state, pcm, 160, &consumed);
    if (consumed != 160 || (result.size && result.size != 40)) return -1;
    if (result.size) std::copy_n(result.values, result.size, out);
    return static_cast<int>(result.size);
}

extern "C" int stream_events(Stream* stream, uint8_t baseline, uint8_t alternate) {
    auto events = stream->decoder.update(baseline, alternate);
    return events.detected | (events.emitted << 2) | (events.suppressed << 4);
}
