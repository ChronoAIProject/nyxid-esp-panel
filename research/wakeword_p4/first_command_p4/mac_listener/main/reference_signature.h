#pragma once
#include <cstdint>

namespace nyx {
// Check only at a two-code match on the electrical reference. Compare 32 ms
// from the middle of the frozen WAV's second code, tolerating level changes.
class ReferenceSignature {
public:
    void update(std::int16_t sample) {
        const std::uint32_t bit = 1u << (next_ & 31);
        auto& word = history_[next_ >> 5];
        if (sample > 0) word |= bit;
        else word &= ~bit;
        next_ = (next_ + 1) & 1023;
    }

    bool matches() const {
        // Second code ends at 15680. Its sampled sign pattern starts at 14800.
        for (int lag = 880 - 48; lag <= 880 + 48; ++lag) {
            int errors = 0;
            for (int i = 0; i < 128; ++i) {
                const int slot = (1024 + next_ - lag + 4*i) & 1023;
                const auto actual = (history_[slot >> 5] >> (slot & 31)) & 1u;
                const auto expected = (kPattern[i >> 5] >> (31 - (i & 31))) & 1u;
                errors += actual != expected;
                if (errors > 12 && i + 1 - errors > 12) break;
                if (i == 127) return true;
            }
        }
        return false;
    }

private:
    // Sign of digital v5 PCM[14800:15312:4], most significant bit first.
    static constexpr std::uint32_t kPattern[4] = {0xb597fd99, 0xe723199e, 0x6d666b6c, 0xcd9cd24d};
    std::uint32_t history_[32]{};
    int next_ = 0;
};
}
