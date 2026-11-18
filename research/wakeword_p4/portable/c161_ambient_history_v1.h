#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace nyx {

// Call before the 160-sample MicroFrontend update. One instance belongs to the
// alternate frontend for the lifetime of a complete 16 kHz stream.
class AmbientHistory {
 public:
  void update(bool speaking, std::uint32_t* estimate) {
    if (speaking == speaking_) return;
    if (speaking) {
      std::copy_n(estimate, saved_.size(), saved_.begin());
    } else {
      std::copy(saved_.begin(), saved_.end(), estimate);
    }
    speaking_ = speaking;
  }

 private:
  std::array<std::uint32_t, 40> saved_{};
  bool speaking_ = false;
};

}  // namespace nyx
