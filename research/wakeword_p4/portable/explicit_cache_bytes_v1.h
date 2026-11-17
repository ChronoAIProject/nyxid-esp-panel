#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace nyx {
// Only the frozen B40/history INT8 ABI: caller verifies tensor names/shapes and
// model SHA. Keep seven dedicated retained input buffers per interpreter;
// output views belong to its arena. Never share caches across models.
struct Int8Bytes {
  std::int8_t* data;
  std::size_t size;
  float scale;
  int zero;
};

inline bool valid_int8_bytes(const Int8Bytes& tensor) {
  return tensor.data && tensor.size && std::isfinite(tensor.scale) &&
         tensor.scale > 0 && tensor.zero >= -128 && tensor.zero <= 127;
}

inline bool reset_explicit_cache(const Int8Bytes& cache) {
  if (!valid_int8_bytes(cache)) return false;
  std::memset(cache.data, static_cast<unsigned char>(cache.zero), cache.size);
  return true;
}

// Exact-byte path only. A future ABI needing requantization must be implemented
// and verified separately; silently copying different scales is an error.
inline bool feedback_explicit_cache(const Int8Bytes& input,
                                    const Int8Bytes& output) {
  if (!valid_int8_bytes(input) || !valid_int8_bytes(output) ||
      input.size != output.size || input.scale != output.scale ||
      input.zero != output.zero) return false;
  std::memmove(input.data, output.data, input.size);
  return true;
}

inline bool signed_probability_code(const Int8Bytes& probability,
                                    std::uint8_t& code) {
  if (!valid_int8_bytes(probability) || probability.size != 1 ||
      probability.scale != 1.f/256 || probability.zero != -128) return false;
  code = static_cast<std::uint8_t>(static_cast<int>(probability.data[0]) + 128);
  return true;
}
}  // namespace nyx
