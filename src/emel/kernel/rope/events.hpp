#pragma once

#include <cstdint>
#include <span>

namespace emel::kernel::rope::event {

struct dispatch_result {
  bool accepted = false;
};

// Precomputes cos/sin rotation tables for positions [0, positions):
// freqs[i] = theta^(-2i / head_dim) for i in [0, head_dim/2);
// cos_out[p * head_dim/2 + i] = cos(p * freqs[i]), sin likewise. Matches the
// reference `precompute_rope_freqs`.
struct precompute_request {
  float theta = 0.0f;
  uint32_t head_dim = 0u;
  uint32_t positions = 0u;
  std::span<float> cos_out;
  std::span<float> sin_out;
};

// Applies the interleaved-half rotation from the reference `apply_rope` to
// `head_count` contiguous head rows of `head_dim` floats each, in place:
// x1 = row[:half], x2 = row[half:];
// row = [x1*cos - x2*sin, x2*cos + x1*sin] with the `position` table row.
struct apply_rows_request {
  std::span<const float> cos_table;
  std::span<const float> sin_table;
  uint32_t position = 0u;
  uint32_t head_count = 0u;
  uint32_t head_dim = 0u;
  std::span<float> rows;
};

struct execute_precompute {
  const precompute_request &request;
  dispatch_result &result;
};

struct execute_apply_rows {
  const apply_rows_request &request;
  dispatch_result &result;
};

} // namespace emel::kernel::rope::event
