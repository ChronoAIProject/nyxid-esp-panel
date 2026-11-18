#include "../portable/c161_pcm_cue_matcher_v1.h"
#include <new>

extern "C" nyx::PcmCueMatcher* make_pcm_cue() { return new (std::nothrow) nyx::PcmCueMatcher{}; }
extern "C" void destroy_pcm_cue(nyx::PcmCueMatcher* cue) { delete cue; }
extern "C" int pcm_cue_bytes() { return sizeof(nyx::PcmCueMatcher); }
extern "C" int pcm_cue_block(nyx::PcmCueMatcher* cue, const std::int16_t* pcm, int count,
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
