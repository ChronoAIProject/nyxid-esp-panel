#pragma once

#include <cstdint>
#include <optional>
#include "c161_coded_cue_reference_v5.h"

namespace nyx {
// The digital playback stream may contain two complete triples in one utterance.
// Rearm only its matcher after each confirmed triple. Keep the original echo window.
class RepeatedCueReference {
public:
    bool allow(std::int16_t reference, bool mic_hit) {
        if (matcher_->update(reference)) {
            remaining_ = CodedCueReferenceV5::kWindowSamples + 1;
            matcher_.emplace();
        }
        const bool clear = remaining_ == 0;
        if (!clear) --remaining_;
        return clear && mic_hit;
    }

private:
    std::optional<CodedCueV5> matcher_{std::in_place};
    int remaining_ = 0;
};
}
