// Synthetic replay with the host wheel's existing MicroFrontend entry points.
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

extern "C" int ambient_history(const int16_t *pcm, const uint8_t *speech, size_t hops,
                               int treatment, uint16_t *output, size_t capacity) {
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
    FrontendState state{};
    if (!FrontendPopulateState(&config, &state, 16000)) {
        FrontendFreeStateContents(&state);
        return -1;
    }
    std::array<uint32_t, 40> ambient{};
    bool was_speech = false;
    size_t count = 0;
    for (size_t hop = 0; hop < hops; ++hop) {
        bool speaking = speech[hop] != 0;
        auto *estimate = state.noise_reduction.estimate;
        if (treatment && speaking != was_speech) {
            if (speaking) {
                std::copy_n(estimate, ambient.size(), ambient.begin());
            } else {
                std::copy(ambient.begin(), ambient.end(), estimate);
            }
        }
        was_speech = speaking;
        size_t used = 0;
        FrontendOutput frame = FrontendProcessSamples(&state, pcm + 160 * hop, 160, &used);
        if (used != 160 || (frame.size && frame.size != 40) || count + frame.size > capacity) {
            FrontendFreeStateContents(&state);
            return -2;
        }
        if (frame.size) {
            std::copy_n(frame.values, frame.size, output + count);
            count += frame.size;
        }
    }
    FrontendFreeStateContents(&state);
    return count == capacity ? 0 : -3;
}
