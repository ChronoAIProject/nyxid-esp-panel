#pragma once
#include "c161_binary40_events_v1.h"

namespace nyx {
// Bits 1/2/4 identify original C161/Binary40/history treatment respectively.
// Independent detector histories; no maximum or blending of raw scores.
class TripleEvents {
 public:
  Events update(std::uint8_t c161, std::uint8_t binary40,
                std::uint8_t history) {
    ++step_;
    const std::uint8_t detected = (c161_.update(step_, c161) ? 1 : 0)
        | (binary40_.update(step_, binary40) ? 2 : 0)
        | (history_.update(step_, history) ? 4 : 0);
    std::uint8_t emitted = 0;
    if (detected && step_ - last_emitted_ >= 67) {
      emitted = (detected & 1) ? 1 : ((detected & 2) ? 2 : 4);
      last_emitted_ = step_;
    }
    return {detected, emitted, static_cast<std::uint8_t>(detected & ~emitted)};
  }
 private:
  Detector c161_, binary40_, history_;
  std::int64_t step_ = 0, last_emitted_ = -1000;
};
// Construct only at full-stream start; feed every prime/audio/tail call.
// Suppression does not reset a detector or extend shared emitted cooldown.
// Binary40 and history signed probability outputs need +128 before update.
}  // namespace nyx
