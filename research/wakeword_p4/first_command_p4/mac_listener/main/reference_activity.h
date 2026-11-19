#pragma once
#include <cstdint>

namespace nyx {
// Emit one activity pulse at the end of each 10 ms frame whose AC RMS reaches
// 64 PCM16 units. The wake and capture routes apply their existing hold times.
class ReferenceActivity {
public:
    bool update(int16_t sample) {
        sum_ += sample;
        energy_ += int(sample) * int(sample);
        if (++count_ != kFrame) return false;
        const bool active = kFrame * energy_ - int64_t(sum_) * sum_ >=
                            int64_t(kFrame * 64) * (kFrame * 64);
        count_ = sum_ = 0;
        energy_ = 0;
        return active;
    }

private:
    static constexpr int kFrame = 160;
    int count_ = 0, sum_ = 0;
    int64_t energy_ = 0;
};
}
