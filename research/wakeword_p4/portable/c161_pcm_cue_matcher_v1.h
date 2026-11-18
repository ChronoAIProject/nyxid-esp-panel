#pragma once
#include <algorithm>
#include <cstdint>
#include "c161_pcm_cue_coefficients_v1.h"

namespace nyx {
// Streaming correlation against two specific distributed PCM recordings.
// Return bits 0/1 identify v1/v2. Feed every native sample in order.
class PcmCueMatcher {
public:
    std::uint8_t update(std::int16_t pcm) {
        sum_ += pcm;
        if (++phase_ != 8) return 0;
        const int mean = sum_ / 8;
        const int x = std::clamp((16 * mean - 14 * previous_) / 256, -2047, 2047);
        sum_ = phase_ = 0;
        previous_ = mean;
        const int old = samples_[cursor_];
        energy_ += x*x - old*old;
        samples_[cursor_] = samples_[cursor_+160] = static_cast<std::int16_t>(x);
        cursor_ = (cursor_+1) % 160;
        ++step_;
        const int slot = static_cast<unsigned>(step_) & 1023;
        std::uint8_t hit = 0;
        for (int k=0; k<2; ++k) {
            const int delay = kAnchorDelays[k];
            // Keep counts for both signs in the first-anchor interval.
            const int entering = (slot - delay + 24 + 1024) & 1023;
            const int leaving = (slot - delay - 25 + 1024) & 1023;
            const int add = signs_[k][entering], remove = signs_[k][leaving];
            positive_[k] += (add > 0) - (remove > 0);
            negative_[k] += (add < 0) - (remove < 0);
            signs_[k][slot] = sign(k, 0);
            if (!positive_[k] && !negative_[k]) continue;
            const int second = sign(k, 1);
            if ((second > 0 && positive_[k]) || (second < 0 && negative_[k]))
                hit |= 1u << k;
        }
        if (hit && step_ - last_ >= 4020) {
            last_ = step_;
            return hit;
        }
        return 0;
    }
private:
    int sign(int cue, int anchor) const {
        if (!energy_) return 0;
        const auto* ref = kWaveAnchors[cue][anchor];
        const auto* audio = samples_ + cursor_;
        int dot = 0;
        for (int i=0; i<160; ++i) dot += audio[i]*ref[i];
        const double threshold = (0.68 * 0.68) * kAnchorEnergy[cue][anchor] * energy_;
        if (static_cast<double>(dot) * dot < threshold) return 0;
        return dot > 0 ? 1 : -1;
    }
    std::int16_t samples_[320]{};
    std::int8_t signs_[2][1024]{};
    int positive_[2]{}, negative_[2]{};
    int phase_=0, sum_=0, previous_=0, cursor_=0, energy_=0;
    std::int64_t step_=0, last_=-4020;
};
} // namespace nyx
