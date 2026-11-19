#pragma once

namespace nyx {
// Call once per 16 kHz capture sample on the listener thread. The optional
// speaker event is the panel demonstration's B,C marker pair on the playout
// clock, mapped to this capture clock. A queued or cancelled playback request
// alone supplies no event. Keep feeding the physical R to the AEC and FIFO.
class PanelDemoGuard {
public:
    bool allow_model(bool reference_pair, bool played_pair) {
        if (reference_pair || played_pair) remaining_ = kTailSamples + 1;
        if (remaining_ == 0) return true;
        --remaining_;
        return false;
    }
    static constexpr int kTailSamples = 1944 + 3488;

private:
    int remaining_ = 0;
};
}  // namespace nyx
