#include "c161_coded_cue_v5.h"

namespace {
nyx::CodedCueV5 detector;
}

extern "C" int cue_step(int16_t pcm)
{
    return detector.update(pcm);
}
