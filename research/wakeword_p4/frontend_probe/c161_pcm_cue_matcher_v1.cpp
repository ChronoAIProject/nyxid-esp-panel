#ifdef NYX_CONFIRM_CUE_SUFFIX
#include "../portable/c161_cue_suffix_v1.h"
using Matcher = nyx::ConfirmedPcmCue;
#else
#include "../portable/c161_pcm_cue_matcher_v1.h"
using Matcher = nyx::PcmCueMatcher;
#endif
#include <new>

extern "C" Matcher* make_pcm_cue() { return new (std::nothrow) Matcher{}; }
extern "C" void destroy_pcm_cue(Matcher* cue) { delete cue; }
extern "C" int pcm_cue_bytes() { return sizeof(Matcher); }
extern "C" int pcm_cue_block(Matcher* cue, const std::int16_t* pcm, int count,
                             std::int32_t* ends, std::uint8_t* ids) {
    int emitted = 0;
    for (int i=0; i<count; ++i) {
        const auto hit = cue->update(pcm[i]);
        if (hit) {
            ends[emitted] = i + 1;
            ids[emitted++] = hit;
        }
    }
    return emitted;
}
