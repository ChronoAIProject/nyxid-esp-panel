// The Python driver pipes little-endian PCM16 frames from one unmodified capture.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "c161_coded_cue_v5.h"
#include "c161_coded_cue_aligned_v4.h"
#include "reference_signature.h"
#include "reference_activity.h"

struct Levels {
    std::uint64_t count = 0, full_scale = 0, at_least_64 = 0;
    int peak = 0;
    double squared_sum = 0;

    void update(int value) {
        const int magnitude = std::abs(value);
        if (magnitude > peak) peak = magnitude;
        ++count;
        full_scale += value == -32768 || value == 32767;
        at_least_64 += magnitude >= 64;
        squared_sum += double(value) * value;
    }

    void print(const char *label) const {
        if (count) std::printf("LEVEL %s %llu %d %.8f %llu %llu\n", label,
            static_cast<unsigned long long>(count), peak, std::sqrt(squared_sum / count),
            static_cast<unsigned long long>(at_least_64), static_cast<unsigned long long>(full_scale));
    }
};

int pcm16(const unsigned char *bytes) {
    const int value = bytes[0] + 256 * bytes[1];
    return value < 32768 ? value : value - 65536;
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    const int channels = std::atoi(argv[1]);
    if (channels != 2 && channels != 4) return 1;
    const auto idle = std::strtoull(argv[2], nullptr, 10);
    const std::size_t width = 2 * channels;
    std::array<unsigned char, 32768> buffer{};
    std::uint64_t clock = 0;
    Levels mic_levels, ref_levels, mic_idle, ref_idle;
    nyx::CodedCueV5 mic_match, ref_match;
    nyx::AlignedCodedCueV4 pair;
    nyx::ReferenceSignature signature;
    nyx::ReferenceActivity activity;
    std::uint64_t active_frames = 0, idle_active_frames = 0;
    while (const auto bytes = std::fread(buffer.data(), 1, buffer.size(), stdin)) {
        if (bytes % width) return 1;
        for (std::size_t i = 0; i < bytes; i += width) {
            const int mic = pcm16(&buffer[i]), ref = pcm16(&buffer[i+2]);
            mic_levels.update(mic);
            ref_levels.update(ref);
            if (++clock <= idle) {
                mic_idle.update(mic);
                ref_idle.update(ref);
            }
            signature.update(ref);
            if (activity.update(ref)) {
                ++active_frames;
                if (clock <= idle) ++idle_active_frames;
            }
            const auto at = static_cast<unsigned long long>(clock);
            if (mic_match.update(mic)) std::printf("EVENT mic_three %llu\n", at);
            if (ref_match.update(ref)) std::printf("EVENT ref_three %llu\n", at);
            if (pair.update(ref)) {
                std::printf("EVENT ref_two %llu\n", at);
                if (signature.matches()) std::printf("EVENT ref_signed_two %llu\n", at);
            }
        }
    }
    if (std::ferror(stdin) || !clock || clock < idle) return 1;
    mic_levels.print("mic");
    ref_levels.print("ref");
    mic_idle.print("mic_idle");
    ref_idle.print("ref_idle");
    std::printf("ACTIVITY ref %llu %llu\n", static_cast<unsigned long long>(active_frames),
                static_cast<unsigned long long>(clock / 160));
    if (idle) std::printf("ACTIVITY ref_idle %llu %llu\n", static_cast<unsigned long long>(idle_active_frames),
                          static_cast<unsigned long long>(idle / 160));
}
