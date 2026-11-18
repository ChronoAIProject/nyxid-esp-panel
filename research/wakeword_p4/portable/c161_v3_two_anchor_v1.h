#pragma once

#include <algorithm>
#include <cstdint>

#include "c161_cue_suffix_coefficients_v1.h"
#include "c161_pcm_cue_coefficients_v1.h"

namespace nyx {
// Match either v3 vowel anchor, followed by the recording's /ks/ at the expected delay.
// Feed all consecutive 16 kHz samples. A true return is timestamped at the current sample.
class V3TwoAnchor {
public:
    bool update(std::int16_t pcm) {
        pair_sum_ += pcm;
        if (++phase_ & 1) return false;
        const int mean = pair_sum_ / 2;
        const int high = std::clamp((mean - previous_pair_) / 16, -2047, 2047);
        previous_pair_ = mean;
        low_sum_ += pair_sum_;
        pair_sum_ = 0;

        const int old = high_history_[high_cursor_];
        high_energy_ += high * high - old * old;
        high_history_[high_cursor_] = high_history_[high_cursor_ + 1024] = high;
        high_cursor_ = (high_cursor_ + 1) & 1023;
        if (phase_ == 8) {
            phase_ = 0;
            update_vowels();
        }

        if (cooldown_ > 0) {
            --cooldown_;
            return false;
        }
        if (!pending_[0] && !pending_[1]) return false;
        if (!high_energy_) return false;
        std::int64_t dot = 0;
        for (int i = 0; i < 1024; ++i)
            dot += high_history_[high_cursor_ + i] * kSuffixes[1][i];
        if (double(dot) * dot < 0.04 * kSuffixEnergy[1] * high_energy_) return false;
        cooldown_ = 16079;  // Next admissible report after 32160 native samples.
        return true;
    }

private:
    void update_vowels() {
        const int mean = low_sum_ / 8;
        low_sum_ = 0;
        const int x = std::clamp((16 * mean - 14 * previous_low_) / 256, -2047, 2047);
        previous_low_ = mean;
        const int old = low_history_[low_cursor_];
        low_history_[low_cursor_] = low_history_[low_cursor_ + 160] = x;
        low_energy_ += x * x - old * old;
        low_cursor_ = (low_cursor_ + 1) % 160;
        slot_ = (slot_ + 1) & 1023;
        std::uint8_t mask = 0;
        for (int k = 0; k < 2; ++k) {
            const int delay = kSuffixDelays[1] / 4 + (k == 0 ? kAnchorDelays[1] : 0);
            const int add = (slot_ - delay + 32 + 1024) & 1023;
            const int remove = (slot_ - delay - 33 + 1024) & 1023;
            pending_[k] += int((vowels_[add] >> k) & 1) - int((vowels_[remove] >> k) & 1);
            if (!low_energy_) continue;
            int dot = 0;
            for (int i = 0; i < 160; ++i)
                dot += low_history_[low_cursor_ + i] * kWaveAnchors[1][k][i];
            if (double(dot) * dot >= 0.25 * kAnchorEnergy[1][k] * low_energy_)
                mask |= 1u << k;
        }
        vowels_[slot_] = mask;
    }

    std::int16_t high_history_[2048]{}, low_history_[320]{};
    std::uint8_t vowels_[1024]{};
    std::int64_t high_energy_ = 0, low_energy_ = 0;
    int phase_ = 0, pair_sum_ = 0, previous_pair_ = 0, low_sum_ = 0, previous_low_ = 0;
    int high_cursor_ = 0, low_cursor_ = 0, slot_ = 0, pending_[2]{}, cooldown_ = 0;
};
}  // namespace nyx
