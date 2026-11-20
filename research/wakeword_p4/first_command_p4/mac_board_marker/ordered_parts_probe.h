#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "ordered_parts_weights.h"

namespace ordered_parts {
constexpr int missing = -100000000;
static unsigned pending_tick;
static int32_t pending_scores[2];

struct History {
    int32_t earlier[10];
    int32_t pair[10];
    int32_t previous = missing;
    History() {
        std::fill_n(earlier,10,missing);
        std::fill_n(pair,10,missing);
    }
    int32_t observe(const Weights &w, const int32_t parts[3], unsigned tick) {
        const unsigned now = tick % 10;
        int32_t a = missing, b = missing;
        for (unsigned d = w.minimum1; d <= w.maximum1; ++d)
            a = std::max(a, earlier[(now+10-d)%10]);
        for (unsigned d = w.minimum2; d <= w.maximum2; ++d)
            b = std::max(b, pair[(now+10-d)%10]);
        earlier[now] = parts[0];
        pair[now] = a+parts[1];
        const int32_t total = b+parts[2];
        const int32_t score = std::min(previous,total);
        previous = total;
        return score;
    }
};

struct Probe {
    float pool[20]{};
    uint8_t patch[100]{};
    History history[2];
    unsigned tick = 0;

    void frame(const uint16_t *features, unsigned index) {
        for (int b=0; b<20; ++b) {
            const float left = std::min(features[2*b]/25.6f,26.f);
            const float right = std::min(features[2*b+1]/25.6f,26.f);
            pool[b] += left+right;
        }
        if (index != 2) return;
        std::memmove(patch,patch+20,80);
        for (int b=0; b<20; ++b) {
            patch[80+b] = static_cast<uint8_t>(std::nearbyint((pool[b]/6.f)*(255.f/26.f)));
            pool[b] = 0.f;
        }
        int32_t scores[2];
        const Weights *models[] = {&fixed, &variable};
        for (int m=0; m<2; ++m) {
            const Weights &w = *models[m];
            int32_t parts[] = {w.b2[0],w.b2[1],w.b2[2]};
            for (int j=0; j<16; ++j) {
                int32_t sum = w.b1[j];
                for (int k=0; k<100; ++k) sum += patch[k]*w.w1[j*100+k];
                int32_t hidden = std::clamp<int32_t>((sum+32)/64,0,255);
                for (int k=0; k<3; ++k) parts[k] += hidden*w.w2[16*k+j];
            }
            scores[m] = history[m].observe(w,parts,tick);
        }
        ++tick;
        // First physical 480-sample model read with real audio ends at tick 170.
        if (tick >= 170) {
            pending_tick = tick;
            std::copy_n(scores, 2, pending_scores);
        }
    }
};
}

// The fixture calls this after restoring task interrupts, when UART I/O is allowed.
extern "C" void report_ordered_parts(void) {
    using namespace ordered_parts;
    if (pending_tick) {
        printf("PART %u %ld %ld\n", pending_tick, (long)pending_scores[0], (long)pending_scores[1]);
        pending_tick = 0;
    }
}
