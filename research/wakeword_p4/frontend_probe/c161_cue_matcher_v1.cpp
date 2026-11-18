#include "../portable/c161_cue_matcher_v1.h"
#include <array>
#include <new>

extern "C" {
struct Stream;
Stream* make_stream();
int stream_hop(Stream*, const std::int16_t*, int, std::uint16_t*);
void destroy_stream(Stream*);
}

struct CueStream {
    Stream* frontend = make_stream();
    nyx::CueMatcher matcher{};
    int pending = 0;
    ~CueStream() { if (frontend) destroy_stream(frontend); }
};

extern "C" CueStream* make_cue() {
    auto* s = new (std::nothrow) CueStream{};
    if (s && !s->frontend) { delete s; return nullptr; }
    return s;
}

extern "C" void destroy_cue(CueStream* s) { delete s; }
extern "C" int cue_state_bytes() { return sizeof(nyx::CueMatcher); }

// Input is a sequence of complete 160-sample hops, padded only at true EOF.
// The caller reserves at least hops/3+2 output slots per call.
extern "C" int cue_block(CueStream* s, const std::int16_t* pcm, int hops, bool final,
                         float* scores, std::uint8_t* events) {
    std::array<std::uint16_t,40> feature{};
    int count = 0;
    auto submit = [&]() {
        auto output = s->matcher.update(feature.data());
        s->pending = (s->pending + 1) % 3;
        if (output.ready) {
            scores[2*count] = output.scores[0];
            scores[2*count+1] = output.scores[1];
            events[count++] = output.event;
        }
    };
    for (int i = 0; i < hops; ++i) {
        const int size = stream_hop(s->frontend, pcm+160*i, 0, feature.data());
        if (size < 0) return -1;
        if (size) submit();
    }
    if (final) {
        feature.fill(0);
        while (s->pending) submit();
    }
    return count;
}
