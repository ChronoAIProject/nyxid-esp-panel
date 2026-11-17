#pragma once
#include <cstdint>

namespace nyx {
// Feed the exact uint8 probability code, once per native 480-sample call.
// Binary40 signed output must first be translated with code = signed + 128.
class Detector {
 public:
  bool update(std::int64_t step, std::uint8_t code) {
    if (count_ == 5) total_ -= history_[cursor_];
    else ++count_;
    history_[cursor_] = code;
    total_ += code;
    cursor_ = (cursor_ + 1) % 5;
    if (count_ < 5) return false;
    if (2 * total_ < 1175 && step - last_ >= 67) armed_ = true;
    if (armed_ && total_ >= 1175 && step - last_ >= 67) {
      last_ = step;
      armed_ = false;
      return true;
    }
    return false;
  }
 private:
  std::uint8_t history_[5]{};
  int cursor_ = 0, count_ = 0, total_ = 0;
  std::int64_t last_ = -1000;
  bool armed_ = true;
};

struct Events {
  std::uint8_t detected;   // bit 0 C161, bit 1 Binary40
  std::uint8_t emitted;    // 0, 1 (C161), or 2 (Binary40)
  std::uint8_t suppressed;
};

class ParallelEvents {
 public:
  Events update(std::uint8_t c161, std::uint8_t binary40) {
    ++step_;
    const std::uint8_t detected = (c161_.update(step_, c161) ? 1 : 0)
        | (binary40_.update(step_, binary40) ? 2 : 0);
    std::uint8_t emitted = 0;
    if (detected && step_ - last_emitted_ >= 67) {
      emitted = (detected & 1) ? 1 : 2;  // C161 wins simultaneous events.
      last_emitted_ = step_;
    }
    return {detected, emitted, static_cast<std::uint8_t>(detected & ~emitted)};
  }
 private:
  Detector c161_, binary40_;
  std::int64_t step_ = 0, last_emitted_ = -1000;
};
// Construct a new ParallelEvents only at complete stream start. Feed all prime
// and tail calls, and keep both model caches and this object across prime/audio
// boundaries. A suppressed event never changes either detector or cooldown.
}  // namespace nyx
