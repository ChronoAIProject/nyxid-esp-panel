#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace nyx {
// One update per emitted 16 kHz sample, using the mic and playback reference
// at that output index. The dry lane follows the measured ESP-SR signal delay.
class ReferenceCapture {
public:
    int16_t update(int16_t mic, int16_t reference, int16_t processed) {
        return update_active(mic, std::abs(int(reference)) >= 64, processed);
    }

    int16_t update_active(int16_t mic, bool active, int16_t processed) {
        const int dry = delay_[cursor_];
        delay_[cursor_] = mic;
        cursor_ = (cursor_ + 1) % kDelay;
        if (active) hold_ = 8001;
        if (hold_) {
            --hold_;
            weight_ = std::min(weight_ + 1, kFade);
        } else {
            weight_ = std::max(weight_ - 1, 0);
        }
        return (dry * (kFade - weight_) + int(processed) * weight_) / kFade;
    }

private:
    static constexpr int kDelay = 1440, kFade = 160;
    int16_t delay_[kDelay]{};
    int cursor_ = 0, hold_ = 0, weight_ = 0;
};
}
