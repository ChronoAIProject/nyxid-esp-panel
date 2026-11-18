#if defined(NYX_ALIGNED_CODED_CUE_V4)
#include "../portable/c161_coded_cue_aligned_v4.h"
using Detector = nyx::AlignedCodedCueV4;
constexpr unsigned char kBit = 4;
#elif defined(NYX_BAND_CODED_CUE_V4)
#include "../portable/c161_coded_cue_band_v4.h"
using Detector = nyx::BandCodedCueV4;
constexpr unsigned char kBit = 4;
#elif defined(NYX_CODED_CUE_V4)
#include "../portable/c161_coded_cue_v4.h"
using Detector = nyx::CodedCueV4;
constexpr unsigned char kBit = 4;
#else
#include "../portable/c161_v3_two_anchor_v1.h"
using Detector = nyx::V3TwoAnchor;
constexpr unsigned char kBit = 2;
#endif

extern "C" {
void* make_pcm_cue() { return new Detector(); }
void destroy_pcm_cue(void* state) { delete static_cast<Detector*>(state); }
int pcm_cue_bytes() { return sizeof(Detector); }
int pcm_cue_block(void* state, const std::int16_t* pcm, int length, int* ends, std::uint8_t* bits) {
    auto* detector = static_cast<Detector*>(state);
    int count = 0;
    for (int i = 0; i < length; ++i) {
        if (detector->update(pcm[i])) {
            ends[count] = i + 1;
            bits[count++] = kBit;
        }
    }
    return count;
}
}
