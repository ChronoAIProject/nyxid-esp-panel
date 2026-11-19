#include "c161_coded_cue_v5.h"
#ifdef NYX_SPEEX_COLD_GUARD
#include "c161_coded_cue_reference_v5.h"

extern "C" int reference_allows(int16_t reference, int mic_hit)
{
    static nyx::CodedCueReferenceV5 guard;
    return guard.allow(reference, mic_hit);
}
#endif

namespace {
nyx::CodedCueV5 detector;
}

extern "C" int cue_step(int16_t pcm)
{
    return detector.update(pcm);
}
