#pragma once
#include <cstdlib>
#include <cstdint>

namespace nyx {
// A microphone decision can arrive shortly before the playback reference
// completes its three v5 anchors. Allow one C161 stride during playback.
class ReferenceWake {
public:
    bool update(int16_t reference, bool reference_clear, bool hit, bool awaiting_third = false) {
        if (std::abs(int(reference)) >= 64) recent_ = 8001;
        const bool playing = recent_ > 0;
        if (playing) --recent_;
        if (!reference_clear) {
            pending_ = 0;
            return false;
        }
        if (pending_) {
            // Keep a due decision until a tentative pair expires or a full match cancels it.
            if (pending_ == 1 && awaiting_third) return false;
            if (--pending_ == 0) return true;
        }
        if (!hit) return false;
        if (playing) pending_ = 480;
        else if (awaiting_third) pending_ = 1;
        else return true;
        return false;
    }

private:
    int recent_ = 0, pending_ = 0;
};
}
