#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#if defined(NYX_CUE_SELFTEST_V5)
#include "c161_coded_cue_v5.h"
#else
#include "c161_cue_suffix_v1.h"
#endif
#include "esp_heap_caps.h"
#include "esp_timer.h"

namespace {
constexpr int prime = 81600;
constexpr int block = 480;
#if defined(NYX_CUE_SELFTEST_V5)
extern const std::uint8_t cue_start[] asm("_binary_4090_c161_coded_cue_v5_command_far_wav_start");
extern const std::uint8_t cue_end[] asm("_binary_4090_c161_coded_cue_v5_command_far_wav_end");
constexpr int length = 142941;
constexpr int gap = 32163;  // Preserve the 4-sample phase on the second replay.
constexpr int first_event = 102932;
constexpr unsigned expected_bits = 8;
nyx::CodedCueV5 matcher;
unsigned detect(std::int16_t pcm) { return matcher.update(pcm) ? expected_bits : 0; }
#else
extern const std::uint8_t cue_start[] asm("_binary_hye_nyx_c161_synthetic_cue_v3_wav_start");
extern const std::uint8_t cue_end[] asm("_binary_hye_nyx_c161_synthetic_cue_v3_wav_end");
constexpr int length = 21200;
constexpr int gap = 32160;
constexpr int first_event = 13214;
constexpr unsigned expected_bits = 2;
nyx::ConfirmedPcmCue matcher;
unsigned detect(std::int16_t pcm) { return matcher.update(pcm); }
#endif

std::int16_t input(int t) {
    if (t < 0) return 0;
    if (t >= length + gap) t -= length + gap;
    if (t >= length) return 0;
    std::int16_t pcm;
    std::memcpy(&pcm, cue_start + 44 + 2*t, sizeof(pcm));
    return pcm;
}
} // namespace

extern "C" void app_main() {
    if (cue_end - cue_start != 44 + length*2 || std::memcmp(cue_start+36, "data", 4)) {
        std::puts("NYX_CUE_SELFTEST FAIL fixture");
        return;
    }
    constexpr int expected[] = {first_event, first_event + length + gap};
    int hits = 0, overruns = 0;
    std::int64_t maximum = 0;
    bool matches = true;
    for (int start = -prime; start < 2*length+gap; start += block) {
        int positions[block/8+1], count = 0;
        std::uint8_t ids[block/8+1];
        const auto before = esp_timer_get_time();
        for (int t=start; t<std::min(start+block,2*length+gap); ++t) {
            const auto hit = detect(input(t));
            if (hit) {
                positions[count] = t+1;
                ids[count++] = hit;
            }
        }
        const auto us = esp_timer_get_time() - before;
        maximum = std::max(maximum,us);
        if (us >= 30000) ++overruns;
        for (int i=0; i<count; ++i) {
            std::printf("NYX_CUE_EVENT sample=%d bits=%u\n", positions[i], unsigned(ids[i]));
            matches &= hits < 2 && positions[i] == expected[hits] && ids[i] == expected_bits;
            ++hits;
        }
    }
    std::printf("NYX_CUE_SELFTEST %s hits=%d state=%u max_block_us=%" PRId64
                " blocks_ge_30ms=%d min_internal_heap=%u\n",
                matches && hits==2 && overruns==0 ? "PASS" : "FAIL", hits,
                unsigned(sizeof(matcher)), maximum, overruns,
                unsigned(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
}
