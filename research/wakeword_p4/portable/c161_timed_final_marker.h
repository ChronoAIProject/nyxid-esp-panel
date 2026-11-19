#pragma once
#include <cstdint>
#include "c161_coded_cue_coefficients_v5.h"

namespace nyx {
// Feed consecutive mono PCM16 at 16 kHz. Only the distributed v5 cue carries the three patterns.
class TimedFinalMarker {
public:
    bool update(std::int16_t pcm) {
        sum_ += pcm;
        if (++phase_ < 4) return false;
        const int block = sum_ / 16;
        fir_[fir_cursor_] = fir_[fir_cursor_ + 17] = block;
        fir_cursor_ = (fir_cursor_ + 1) % 17;
        int filtered = 0;
        for (int i = 0; i < 17; ++i)
            filtered += fir_[fir_cursor_ + i] * kCueBandFirV5[16 - i];
        const int sample = filtered / 128;
        sum_ = phase_ = 0;
        const int old = ring_[cursor_];
        energy_ += sample * sample - old * old;
        ring_[cursor_] = ring_[cursor_ + 320] = sample;
        cursor_ = (cursor_ + 1) % 320;
        slot_ = (slot_ + 1) & 1023;
        // Look for the next known pattern 160 ms or 120 ms after its predecessor.
        // A matched chain keeps one polarity, allowing 1.5 ms of relative arrival error.
        for (int k = 0; k < 2; ++k) {
            const int delay = k == 0 ? 640 : 480;
            const int entering = signs_[k][(slot_ + 1024 - delay + 6) & 1023];
            const int leaving = signs_[k][(slot_ + 1024 - delay - 7) & 1023];
            positive_[k] += (entering > 0) - (leaving > 0);
            negative_[k] += (entering < 0) - (leaving < 0);
            signs_[k][slot_] = 0;
        }
        if (cooldown_ > 0) {
            --cooldown_;
            return false;
        }
        if (!energy_) return false;
        signs_[0][slot_] = sign(0);
        if (positive_[0] || negative_[0]) {
            const int second = sign(1);
            if (consistent(0, second)) signs_[1][slot_] = second;
        }
        if (!positive_[1] && !negative_[1]) return false;
        if (!consistent(1, sign(2))) return false;
        cooldown_ = 8039;
        return true;
    }

private:
    bool consistent(int k, int sign) const {
        return (sign > 0 && positive_[k]) || (sign < 0 && negative_[k]);
    }

    int sign(int pattern) const {
        std::int32_t dot = 0;
        for (int i = 0; i < 320; ++i)
            dot += ring_[cursor_ + i] * kCueBandPatternsV5[pattern][i];
        const double squared = double(dot) * dot;
        const int polarity = dot > 0 ? 1 : -1;
        if (squared >= 0.09 * kCueBandEnergyV5[pattern] * energy_) return polarity;
        if (pattern != 2 || squared < 0.04 * kCueBandEnergyV5[pattern] * energy_) return 0;

        // A weak final marker needs a same-polarity middle marker 120 ms earlier.
        // Two 4 kHz ticks allow 0.5 ms for clock drift and peak quantization.
        for (int skew = -2; skew <= 2; ++skew) {
            if (signs_[1][(slot_ + 1024 - 480 + skew) & 1023] == polarity)
                return polarity;
        }
        return 0;
    }
    std::int16_t ring_[640]{}, fir_[34]{};
    int fir_cursor_ = 0;
    std::int8_t signs_[2][1024]{};
    std::int64_t energy_ = 0;
    int sum_ = 0, phase_ = 0, cursor_ = 0, slot_ = 0, positive_[2]{}, negative_[2]{}, cooldown_ = 0;
};
}  // namespace nyx
