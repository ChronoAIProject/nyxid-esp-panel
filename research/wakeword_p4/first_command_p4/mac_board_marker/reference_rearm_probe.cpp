// Stream interleaved PCM16 M/R through the same cue and reference gates as the board.
#include "c161_timed_final_marker.h"
#include "c161_coded_cue_reference_v5.h"
#include "reference_activity.h"
#include "reference_signature.h"
#include "reference_prefix.h"
#include "reference_wake.h"
#include <cstdio>
#include <cstdint>

int main() {
    nyx::TimedFinalMarker microphone;
    nyx::CodedCueReferenceV5 reference;
    nyx::ReferenceActivity activity;
    nyx::ReferenceSignature signature;
    nyx::ReferencePrefix prefix;
    nyx::ReferenceWake route;
    bool last_clear = true;
    auto last_prefix = nyx::ReferencePrefix::State::clear;
    std::int16_t mr[2];
    for (int sample = 1; std::fread(mr, sizeof(mr[0]), 2, stdin) == 2; ++sample) {
        bool raw = microphone.update(mr[0]);
        bool active = activity.update(mr[1]);
        bool clear = reference.allow(mr[1], true);
        signature.update(mr[1]);
        auto state = prefix.update(mr[1], [&] { return signature.matches(); });
        using State = nyx::ReferencePrefix::State;
        bool allowed = route.update_active(active, clear && state != State::confirmed, raw, state == State::awaiting_third);
        if (raw) std::printf("raw %d\n", sample);
        if (clear != last_clear) std::printf("%s %d\n", clear ? "veto_end" : "veto_start", sample);
        if (state != last_prefix) std::printf("%s %d\n", state == State::clear ? "prefix_end" :
                                                state == State::confirmed ? "confirmed" : "awaiting", sample);
        if (allowed) std::printf("allowed %d\n", sample);
        last_clear = clear;
        last_prefix = state;
    }
    return std::ferror(stdin) ? 1 : 0;
}
