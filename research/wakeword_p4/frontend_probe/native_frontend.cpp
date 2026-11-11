// Host-only C ABI for the unmodified upstream frontend. No audio device access.
#include <cstdint>
#include <cstdlib>
extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

extern "C" void *nyx_frontend_create() {
    FrontendConfig config{};
    FrontendFillConfigWithDefaults(&config);
    config.window.size_ms = 30;
    config.window.step_size_ms = 10;
    config.filterbank.num_channels = 40;
    config.filterbank.lower_band_limit = 125.0f;
    config.filterbank.upper_band_limit = 7500.0f;
    config.noise_reduction.smoothing_bits = 10;
    config.noise_reduction.even_smoothing = 0.025f;
    config.noise_reduction.odd_smoothing = 0.06f;
    config.noise_reduction.min_signal_remaining = 0.05f;
    config.pcan_gain_control.enable_pcan = 1;
    config.pcan_gain_control.strength = 0.95f;
    config.pcan_gain_control.offset = 80.0f;
    config.pcan_gain_control.gain_bits = 21;
    config.log_scale.enable_log = 1;
    config.log_scale.scale_shift = 6;
    auto *state = static_cast<FrontendState *>(std::calloc(1, sizeof(FrontendState)));
    if (!state) return nullptr;
    if (!FrontendPopulateState(&config, state, 16000)) {
        FrontendFreeStateContents(state);
        std::free(state);
        return nullptr;
    }
    return state;
}

extern "C" int nyx_frontend_process(void *handle, const int16_t *pcm, uint16_t *out) {
    if (!handle || !pcm || !out) return -1;
    size_t consumed = 0;
    auto result = FrontendProcessSamples(static_cast<FrontendState *>(handle), pcm, 160, &consumed);
    if (consumed != 160 || (result.size != 0 && result.size != 40)) return -1;
    for (size_t i = 0; i < result.size; ++i) out[i] = result.values[i];
    return static_cast<int>(result.size);
}

extern "C" int nyx_frontend_weight(void *handle, int index, int unweight) {
    if (!handle || index < 0) return -1;
    const auto &fb = static_cast<FrontendState *>(handle)->filterbank;
    int n = 0;
    for (int i = 0; i <= fb.num_channels; ++i) {
        const int end = fb.channel_weight_starts[i]+fb.channel_widths[i];
        if (end > n) n = end;
    }
    if (index >= n) return -1;
    return (unweight ? fb.unweights : fb.weights)[index];
}

// Controlled diagnostic intervention in an isolated handle, before any PCM.
extern "C" int nyx_frontend_set_weight_pair(void *handle, int index, int weight, int unweight) {
    if (nyx_frontend_weight(handle, index, 0) < 0 || weight < 0 || unweight < 0 ||
        weight > 4096 || unweight > 4096 || weight+unweight != 4096) return 0;
    auto &fb = static_cast<FrontendState *>(handle)->filterbank;
    fb.weights[index] = static_cast<int16_t>(weight);
    fb.unweights[index] = static_cast<int16_t>(unweight);
    return 1;
}

extern "C" void nyx_frontend_destroy(void *handle) {
    if (handle) {
        FrontendFreeStateContents(static_cast<FrontendState *>(handle));
        std::free(handle);
    }
}
