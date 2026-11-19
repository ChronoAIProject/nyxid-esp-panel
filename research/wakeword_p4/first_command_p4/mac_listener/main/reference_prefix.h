#pragma once
#include "c161_coded_cue_aligned_v4.h"

namespace nyx {
// The first two codes of the digital v5 reference precede the final code by
// 120 ms. Bridge that interval, including the final matcher's 1.5 ms tolerance.
class ReferencePrefix {
public:
    static constexpr int kWindowSamples = 1920 + 24;
    enum class State { clear, confirmed, awaiting_third };

    template <typename Qualify>
    State update(std::int16_t reference, Qualify qualify) {
        if (pair_.update(reference)) {
            state_ = qualify() ? State::confirmed : State::awaiting_third;
            remaining_ = kWindowSamples + 1;
        }
        if (!remaining_) return State::clear;
        --remaining_;
        return state_;
    }

    template <typename Qualify>
    bool clear(std::int16_t reference, Qualify qualify) {
        return update(reference, qualify) != State::confirmed;
    }
private:
    AlignedCodedCueV4 pair_;
    int remaining_ = 0;
    State state_ = State::clear;
};
}
