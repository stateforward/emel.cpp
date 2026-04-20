#pragma once

#include <cstddef>
#include <cstdint>

namespace emel::bench {

inline std::uint64_t checksum_bytes(const std::uint8_t * bytes, const std::size_t count) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < count; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

template <class sample_type>
inline void capture_embedding_case_output(sample_type & sample,
                                          const float * values,
                                          const std::size_t count,
                                          const bool capture_compare_outputs,
                                          const bool capture_anchor_output,
                                          bool & anchor_output_captured) {
  sample.output_checksum = checksum_bytes(
    reinterpret_cast<const std::uint8_t *>(values), count * sizeof(float));
  if (capture_compare_outputs && capture_anchor_output && !anchor_output_captured) {
    sample.output_values.assign(values, values + count);
    anchor_output_captured = true;
  }
}

}  // namespace emel::bench
