#include "c161_binary40_events_v1.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

static bool u32(std::uint32_t& value) {
  unsigned char b[4];
  if (std::fread(b, 1, 4, stdin) != 4) return false;
  value = std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8)
      | (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
  return true;
}
int main() {
  char magic[8]; std::uint32_t rows;
  if (std::fread(magic, 1, 8, stdin) != 8 || std::memcmp(magic, "NYXPAR01", 8)
      || !u32(rows) || rows != 1760) return 2;
  std::uint64_t total = 0;
  for (std::uint32_t row = 0; row < rows; ++row) {
    std::uint32_t calls;
    if (!u32(calls) || !calls) return 3;
    nyx::ParallelEvents policy;
    for (std::uint32_t call = 0; call < calls; ++call) {
      unsigned char b[5];
      if (std::fread(b, 1, 5, stdin) != 5) return 4;
      auto event = policy.update(b[0], b[1]);
      if (event.detected != b[2] || event.emitted != b[3] || event.suppressed != b[4]) {
        std::fprintf(stderr, "Mismatch row=%u call=%u\n", row, call + 1);
        return 5;
      }
    }
    total += calls;
  }
  if (std::getchar() != EOF) return 6;
  std::printf("All %u streams, %llu calls match saved events exactly.\n", rows,
      static_cast<unsigned long long>(total));
  return 0;
}
