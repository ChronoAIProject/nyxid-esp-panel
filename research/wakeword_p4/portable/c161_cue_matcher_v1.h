#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "c161_cue_templates_v1.h"

namespace nyx {
struct CueResult {
    bool ready = false;
    bool event = false;
    float scores[2]{};
};

// Feed each real MicroFrontend feature frame once, including warmup.
class CueMatcher {
public:
    CueResult update(const std::uint16_t* features) {
        for (int i = 0; i < 40; ++i)
            partial_[i / 4] += std::min<int>(features[i], 666);
        if (++frames_ < 3) return {};

        for (int b = 0; b < 10; ++b) {
            const int next = partial_[b] / 12;
            const int old = history_[cursor_][b];
            history_[cursor_][b] = next;
            sums_[b] += next - old;
            squares_ += next * next - old * old;
            partial_[b] = 0;
        }
        frames_ = 0;
        cursor_ = (cursor_ + 1) % 32;
        ++steps_;
        CueResult result{};
        result.ready = true;
        if (steps_ < 32) return result;

        std::int64_t energy = 32LL * squares_;
        for (int s : sums_) energy -= static_cast<std::int64_t>(s) * s;
        if (energy > 0) {
            for (int k = 0; k < 2; ++k) {
                std::int64_t dot = 0;
                for (int t = 0; t < 32; ++t) {
                    const auto& sample = history_[(t + cursor_) % 32];
                    for (int b = 0; b < 10; ++b)
                        dot += static_cast<int>(sample[b]) * kCueTemplates[k][t][b];
                }
                result.scores[k] = static_cast<float>(dot / std::sqrt(energy * kCueEnergy[k]));
            }
        }
        const float score = std::max(result.scores[0], result.scores[1]);
        if (steps_ - last_ >= 67) {
            if (score < 0.55f) armed_ = true;
            if (armed_ && score >= 0.85f) {
                armed_ = false;
                last_ = steps_;
                result.event = true;
            }
        }
        return result;
    }
private:
    std::uint16_t history_[32][10]{};
    int sums_[10]{}, partial_[10]{}, squares_ = 0, frames_ = 0, cursor_ = 0;
    std::int64_t steps_ = 0, last_ = -1000;
    bool armed_ = true;
};
}  // namespace nyx
