#pragma once
#include <cstdint>
#include "c161_coded_cue_band_coefficients_v4.h"

namespace nyx {
// Feed consecutive mono PCM16 at 16 kHz. Only the distributed v4 cue carries both patterns.
class BandCodedCueV4 {
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
        // Expected separation is 160 ms, allowing 8 ms either side for echoes and clock error.
        pending_ += first_[(slot_ + 1024 - 640 + 32) & 1023];
        pending_ -= first_[(slot_ + 1024 - 640 - 33) & 1023];
        first_[slot_] = false;
        if (cooldown_ > 0) {
            --cooldown_;
            return false;
        }
        if (!energy_) return false;
        first_[slot_] = matches(0);
        if (!pending_ || !matches(1)) return false;
        cooldown_ = 8039;
        return true;
    }

private:
    bool matches(int pattern) const {
        std::int64_t dot = 0;
        for (int i = 0; i < 320; ++i)
            dot += ring_[cursor_ + i] * kCueBandPatterns[pattern][i];
        return double(dot) * dot >= 0.09 * kCueBandEnergy[pattern] * energy_;
    }
    std::int16_t ring_[640]{}, fir_[34]{};
    int fir_cursor_ = 0;
    bool first_[1024]{};
    std::int64_t energy_ = 0;
    int sum_ = 0, phase_ = 0, cursor_ = 0, slot_ = 0, pending_ = 0, cooldown_ = 0;
};
}  // namespace nyx
