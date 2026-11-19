#pragma once
#include "c161_panel_demo_guard.h"

namespace nyx {
// Inputs are consumed in capture-sample order. Corroborate the known panel's
// B,C playout event with the same pair heard on the microphone, in either order.
class PanelEchoConfirmation {
public:
    bool allow_model(bool reference_pair, bool played_pair, bool mic_pair) {
        age(played_age_, played_pair);
        age(mic_age_, mic_pair);
        if (reference_pair)
            remaining_ = PanelDemoGuard::kTailSamples + 1;
        if ((played_pair || mic_pair) && played_age_ <= kSkew && mic_age_ <= kSkew) {
            const int tail = PanelDemoGuard::kTailSamples + 1 - mic_age_;
            if (tail > remaining_) remaining_ = tail;
        }
        if (!remaining_) return true;
        --remaining_;
        return false;
    }

private:
    static constexpr int kSkew = 1280;
    static void age(int& value, bool fresh) {
        if (fresh) value = 0;
        else if (value <= kSkew) ++value;
    }
    int played_age_ = kSkew + 1;
    int mic_age_ = kSkew + 1;
    int remaining_ = 0;
};
}  // namespace nyx
