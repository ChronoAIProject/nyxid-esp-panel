#pragma once
#include <cstdint>
#include "c161_coded_cue_coefficients_v5.h"

namespace nyx {
// Acoustic corroboration for a known local B,C,A demonstration. The caller
// separately checks the playout clock; a match here cannot open or veto alone.
class PanelMiddleMarker {
public:
    bool update(std::int16_t pcm) {
        sum_ += pcm;
        if (++phase_ < 4) return false;
        fir_[fir_cursor_] = fir_[fir_cursor_ + 17] = sum_ / 16;
        sum_ = phase_ = 0;
        fir_cursor_ = (fir_cursor_ + 1) % 17;
        int filtered = 0;
        for (int i = 0; i < 17; ++i)
            filtered += fir_[fir_cursor_ + i] * kCueBandFirV5[16 - i];
        const int sample = filtered / 128;
        const int old = ring_[cursor_];
        energy_ += sample * sample - old * old;
        ring_[cursor_] = ring_[cursor_ + 320] = sample;
        cursor_ = (cursor_ + 1) % 320;
        if (!energy_) return false;
        std::int32_t dot = 0;
        for (int i = 0; i < 320; ++i)
            dot += ring_[cursor_ + i] * kCueBandPatternsV5[2][i];
        return double(dot) * dot >= 0.09 * kCueBandEnergyV5[2] * energy_;
    }

private:
    std::int16_t ring_[640]{}, fir_[34]{};
    std::int64_t energy_ = 0;
    int fir_cursor_ = 0, cursor_ = 0, sum_ = 0, phase_ = 0;
};
}  // namespace nyx
