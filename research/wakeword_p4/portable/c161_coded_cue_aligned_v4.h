#pragma once
#include <cstdint>
#include "c161_coded_cue_band_coefficients_v4.h"

namespace nyx {
// Feed consecutive mono PCM16 at 16 kHz. Only the distributed v4 cue carries both patterns.
class AlignedCodedCueV4 {
public:
    bool update(std::int16_t pcm) {
        sum_ += pcm;
        if (++phase_ < 4) return false;
        const int block = sum_ / 16;
        fir_[fir_cursor_] = fir_[fir_cursor_ + 17] = block;
        fir_cursor_ = (fir_cursor_ + 1) % 17;
        int filtered = 0;
        for (int i = 0; i < 17; ++i)
            filtered += fir_[fir_cursor_ + i] * kCueBandFir[16 - i];
        const int sample = filtered / 128;
        sum_ = phase_ = 0;
        const int old = ring_[cursor_];
        energy_ += sample * sample - old * old;
        ring_[cursor_] = ring_[cursor_ + 320] = sample;
        cursor_ = (cursor_ + 1) % 320;
        slot_ = (slot_ + 1) & 1023;
        // The two patterns traverse the same channel. Require the same correlation sign
        // and the 160 ms spacing, allowing 1.5 ms for clock and arrival differences.
        const int entering = first_[(slot_ + 1024 - 640 + 6) & 1023];
        const int leaving = first_[(slot_ + 1024 - 640 - 7) & 1023];
        positive_ += (entering > 0) - (leaving > 0);
        negative_ += (entering < 0) - (leaving < 0);
        first_[slot_] = 0;
        if (cooldown_ > 0) {
            --cooldown_;
            return false;
        }
        if (!energy_) return false;
        first_[slot_] = sign(0);
        if (!positive_ && !negative_) return false;
        const int second = sign(1);
        if (!((second > 0 && positive_) || (second < 0 && negative_))) return false;
        cooldown_ = 8039;
        return true;
    }

private:
    int sign(int pattern) const {
        std::int32_t dot = 0;
        for (int i = 0; i < 320; ++i)
            dot += ring_[cursor_ + i] * kCueBandPatterns[pattern][i];
        if (double(dot) * dot < 0.09 * kCueBandEnergy[pattern] * energy_) return 0;
        return dot > 0 ? 1 : -1;
    }
    std::int16_t ring_[640]{}, fir_[34]{};
    int fir_cursor_ = 0;
    std::int8_t first_[1024]{};
    std::int64_t energy_ = 0;
    int sum_ = 0, phase_ = 0, cursor_ = 0, slot_ = 0, positive_ = 0, negative_ = 0, cooldown_ = 0;
};
}  // namespace nyx
