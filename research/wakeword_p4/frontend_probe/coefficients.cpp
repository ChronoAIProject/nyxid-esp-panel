// Numeric table diagnostic only. No PCM, model, I2S, network or device access.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h"
#include "tensorflow/lite/experimental/microfrontend/lib/window_util.h"
}

template <class T> static void array(const char *key, const T *data, size_t n) {
    std::printf("\"%s\":[", key);
    for (size_t i = 0; i < n; ++i)
        std::printf("%s%lld", i ? "," : "", static_cast<long long>(data[i]));
    std::printf("]");
}

static uint32_t mel_bits(float frequency) {
    const float mel = 1127.0f * log1pf(frequency / 700.0f);
    uint32_t bits;
    std::memcpy(&bits, &mel, sizeof(bits));
    return bits;
}

int main() {
    FilterbankConfig fc{};
    FilterbankFillConfigWithDefaults(&fc);
    fc.num_channels = 40;
    fc.lower_band_limit = 125.0f;
    fc.upper_band_limit = 7500.0f;
    FilterbankState fb{};
    if (!FilterbankPopulateState(&fc, &fb, 16000, 257)) return 1;
    WindowConfig wc{};
    WindowFillConfigWithDefaults(&wc);
    wc.size_ms = 30;
    wc.step_size_ms = 10;
    WindowState window{};
    if (!WindowPopulateState(&wc, &window, 16000)) {
        FilterbankFreeStateContents(&fb);
        return 1;
    }
    size_t weights = 0;
    for (int i = 0; i <= fb.num_channels; ++i) {
        const size_t end = fb.channel_weight_starts[i] + fb.channel_widths[i];
        if (end > weights) weights = end;
    }
    std::printf("{\"schema\":1,\"channels\":%d,\"start_index\":%d,"
                "\"end_index\":%d,\"window_step\":%zu,", fb.num_channels,
                fb.start_index, fb.end_index, window.step);
    array("window_coefficients", window.coefficients, window.size);
    std::printf(",");
    array("channel_frequency_starts", fb.channel_frequency_starts, 41);
    std::printf(",");
    array("channel_weight_starts", fb.channel_weight_starts, 41);
    std::printf(",");
    array("channel_widths", fb.channel_widths, 41);
    std::printf(",");
    array("weights", fb.weights, weights);
    std::printf(",");
    array("unweights", fb.unweights, weights);
    uint32_t mel[259];
    for (int i = 0; i < 257; ++i) mel[i] = mel_bits(i * 31.25f);
    mel[257] = mel_bits(125.0f);
    mel[258] = mel_bits(7500.0f);
    std::printf(",");
    array("diagnostic_mel_float_bits", mel, 259);
    std::printf("}\n");
    WindowFreeStateContents(&window);
    FilterbankFreeStateContents(&fb);
}
