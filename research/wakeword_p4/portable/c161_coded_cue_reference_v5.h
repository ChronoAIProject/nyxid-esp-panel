#pragma once
#include "c161_coded_cue_v5.h"

namespace nyx {
// Feed the raw playback reference at the sample index of the processed mic output.
// The mic matcher can be shared with the existing v5 path. Each result is one tick.
class CodedCueReferenceV5 {
public:
    bool allow(std::int16_t reference, bool mic_hit) {
        if (reference_detector_.update(reference)) {
            remaining_ = kWindowSamples + 1;
        }
        const bool suppress = remaining_ > 0;
        if (suppress) --remaining_;
        return mic_hit && !suppress;
    }

    // Measured zero-reference frontend latency plus eight 256-sample AEC partitions.
    static constexpr int kWindowSamples = 1440 + 8 * 256;
private:
    CodedCueV5 reference_detector_;
    int remaining_ = 0;
};
}
