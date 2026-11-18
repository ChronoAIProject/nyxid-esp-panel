#pragma once
#include "c161_pcm_cue_matcher_v1.h"
#include "c161_cue_suffix_coefficients_v1.h"

namespace nyx {
// Confirms a waveform proposal using its recording's later high-frequency suffix.
class ConfirmedPcmCue {
public:
    std::uint8_t update(std::int16_t pcm) {
        const auto proposal = anchors_.update(pcm);
        sum_ += pcm;
        if (++phase_ != 2) return 0;
        const int mean = sum_/2;
        const int x = std::clamp((mean-previous_)/16, -2047, 2047);
        previous_ = mean;
        sum_ = phase_ = 0;
        const int old = samples_[cursor_];
        samples_[cursor_] = samples_[cursor_+1024] = static_cast<std::int16_t>(x);
        energy_ += x*x - old*old;
        cursor_ = (cursor_ + 1) & 1023;
        ++time_;
        std::uint8_t result = 0;
        for (int k=0; k<2; ++k) {
            if (proposal & (1u << k)) expected_[k] = time_ + kSuffixDelays[k];
            const auto offset = time_ - expected_[k];
            if (offset < -128 || offset > 128 || energy_ == 0) continue;
            std::int64_t dot = 0;
            for (int j=0; j<1024; ++j)
                dot += static_cast<int>(samples_[cursor_+j]) * kSuffixes[k][j];
            if (static_cast<double>(dot)*dot >= 0.04 * kSuffixEnergy[k] * energy_) {
                result |= 1u << k;
                expected_[k] = -1000;
            }
        }
        return result;
    }
private:
    PcmCueMatcher anchors_;
    std::int16_t samples_[2048]{};
    std::int64_t expected_[2]{-1000,-1000}, time_=0, energy_=0;
    int cursor_=0, phase_=0, sum_=0, previous_=0;
};
} // namespace nyx
