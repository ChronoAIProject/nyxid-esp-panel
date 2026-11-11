// Numeric-only embedded PCM. No microphone, I2S, playback, network or NVS calls.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "mbedtls/sha256.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern "C" {
#include "frontend.h"
#include "frontend_util.h"
}

extern const uint8_t vectors_start[] asm("_binary_vectors_bin_start");
extern const uint8_t vectors_end[] asm("_binary_vectors_bin_end");
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Fixture is little endian");

static void require(bool value) { if (!value) std::abort(); }

static void digest(const void *data, size_t n, char *hex) {
    uint8_t result[32];
    require(mbedtls_sha256(static_cast<const uint8_t *>(data), n, result, 0) == 0);
    for (unsigned i = 0; i < 32; ++i) std::sprintf(hex+2*i, "%02x", result[i]);
}

static FrontendConfig configuration() {
    FrontendConfig config{};
    FrontendFillConfigWithDefaults(&config);
    config.window.size_ms = 30; config.window.step_size_ms = 10;
    config.filterbank.num_channels = 40;
    config.filterbank.lower_band_limit = 125.0f; config.filterbank.upper_band_limit = 7500.0f;
    config.noise_reduction.smoothing_bits = 10;
    config.noise_reduction.even_smoothing = 0.025f; config.noise_reduction.odd_smoothing = 0.06f;
    config.noise_reduction.min_signal_remaining = 0.05f;
    config.pcan_gain_control.enable_pcan = 1; config.pcan_gain_control.strength = 0.95f;
    config.pcan_gain_control.offset = 80.0f; config.pcan_gain_control.gain_bits = 21;
    config.log_scale.enable_log = 1; config.log_scale.scale_shift = 6;
    return config;
}

static void tables(const FrontendState &state) {
    const auto &fb = state.filterbank;
    unsigned count = 0;
    for (int i = 0; i <= fb.num_channels; ++i)
        count = std::max(count, unsigned(fb.channel_weight_starts[i]+fb.channel_widths[i]));
    require(count > 293 && state.window.size == 480);
    char window[65], weights[65], unweights[65];
    digest(state.window.coefficients, state.window.size*sizeof(int16_t), window);
    digest(fb.weights, count*sizeof(int16_t), weights);
    digest(fb.unweights, count*sizeof(int16_t), unweights);
    std::printf("P4_FRONTEND_TABLE {\"window_sha256\":\"%s\",\"weights_sha256\":\"%s\","
        "\"unweights_sha256\":\"%s\",\"weight_count\":%u,\"coefficient211\":[%d,%d],"
        "\"coefficient293\":[%d,%d]}\n", window, weights, unweights, count,
        fb.weights[211], fb.unweights[211], fb.weights[293], fb.unweights[293]);
}

static uint32_t read32(const uint8_t *&p, const uint8_t *end) {
    require(size_t(end-p) >= 4);
    uint32_t value; std::memcpy(&value, p, 4); p += 4; return value;
}

extern "C" void app_main() {
    const uint8_t *p = vectors_start;
    require(size_t(vectors_end-p) >= 12 && std::memcmp(p, "NYXFE001", 8) == 0);
    p += 8;
    const unsigned streams = read32(p, vectors_end);
    require(streams == 30);
    const auto config = configuration();
    for (unsigned stream = 0; stream < streams; ++stream) {
        const unsigned index = read32(p, vectors_end), samples = read32(p, vectors_end);
        const unsigned expected_frames = read32(p, vectors_end);
        require(samples >= 480 && samples % 160 == 0 && expected_frames == samples/160-2);
        const size_t pcm_bytes = size_t(samples)*2, raw_bytes = size_t(expected_frames)*80;
        require(size_t(vectors_end-p) >= pcm_bytes+raw_bytes);
        const auto *pcm = reinterpret_cast<const int16_t *>(p); p += pcm_bytes;
        const auto *expected = reinterpret_cast<const uint16_t *>(p); p += raw_bytes;
        FrontendState state{};
        require(FrontendPopulateState(&config, &state, 16000) != 0);
        if (stream == 0) tables(state);
        mbedtls_sha256_context hash;
        mbedtls_sha256_init(&hash);
        require(mbedtls_sha256_starts(&hash, 0) == 0);
        unsigned frames = 0, mismatches = 0, max_delta = 0;
        for (unsigned at = 0; at < samples; at += 160) {
            size_t consumed = 0;
            auto output = FrontendProcessSamples(&state, pcm+at, 160, &consumed);
            require(consumed == 160 && (output.size == 0 || output.size == 40));
            if (!output.size) continue;
            require(frames < expected_frames);
            require(mbedtls_sha256_update(&hash, reinterpret_cast<const uint8_t *>(output.values), 80) == 0);
            for (unsigned c = 0; c < 40; ++c) {
                const unsigned actual = output.values[c], wanted = expected[frames*40+c];
                const unsigned delta = actual > wanted ? actual-wanted : wanted-actual;
                if (delta) {
                    if (mismatches < 8) std::printf("P4_FRONTEND_DELTA {\"pack_index\":%u,"
                        "\"frame\":%u,\"channel\":%u,\"actual\":%u,\"expected\":%u}\n",
                        index, frames, c, actual, wanted);
                    ++mismatches; max_delta = std::max(max_delta, delta);
                }
            }
            ++frames;
        }
        uint8_t bytes[32]; char hex[65];
        require(mbedtls_sha256_finish(&hash, bytes) == 0);
        mbedtls_sha256_free(&hash);
        for (unsigned i = 0; i < 32; ++i) std::sprintf(hex+2*i, "%02x", bytes[i]);
        FrontendFreeStateContents(&state);
        require(frames == expected_frames);
        std::printf("P4_FRONTEND_WITNESS {\"pack_index\":%u,\"frames\":%u,"
            "\"raw_sha256\":\"%s\",\"different_elements\":%u,\"max_abs_delta\":%u}\n",
            index, frames, hex, mismatches, max_delta);
        vTaskDelay(1);
    }
    require(p == vectors_end);
    std::printf("P4_FRONTEND_DONE {\"streams\":%u,\"all_inputs_consumed\":true}\n", streams);
}
