#pragma once
#include <cstdint>

namespace nyx {
// Emit one activity pulse at the end of each 10 ms frame whose AC RMS reaches
// 64 PCM16 units. The wake and capture routes apply their existing hold times.
class ReferenceActivity {
public:
    static constexpr int kFrame = 160;
    static bool active_frame(const int16_t *pcm) {
        int sum = 0;
        int64_t energy = 0;
        for (int i = 0; i < kFrame; ++i) {
            sum += pcm[i];
            energy += int(pcm[i]) * int(pcm[i]);
        }
        return above_floor(sum, energy);
    }

    bool update(int16_t sample) {
        sum_ += sample;
        energy_ += int(sample) * int(sample);
        if (++count_ != kFrame) return false;
        const bool active = above_floor(sum_, energy_);
        count_ = sum_ = 0;
        energy_ = 0;
        return active;
    }

private:
    static bool above_floor(int sum, int64_t energy) {
        return kFrame * energy - int64_t(sum) * sum >= int64_t(kFrame * 64) * (kFrame * 64);
    }
    int count_ = 0, sum_ = 0;
    int64_t energy_ = 0;
};
}
