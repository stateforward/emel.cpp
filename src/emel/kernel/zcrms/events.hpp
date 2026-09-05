#pragma once

#include <cstdint>
#include <span>

namespace emel::kernel::zcrms::event {

struct dispatch_result {
  bool accepted = false;
};

// ZCRMSNorm rows: output[r, i] = (1 + scale[i]) * input[r, i] / rms(input[r]).
// rms uses mean-of-squares with eps = 1e-6 in f32, matching the reference
// `_zcrms`. `input` and `output` may alias (elementwise write per index).
struct norm_rows_request {
  std::span<const float> input;
  std::span<const float> scale;
  uint32_t rows = 0u;
  uint32_t dim = 0u;
  std::span<float> output;
};

// Unit-RMS rows: output[r, i] = input[r, i] * rsqrt(mean(input[r]^2) + eps),
// matching the reference `_rms_unit`.
struct unit_rows_request {
  std::span<const float> input;
  uint32_t rows = 0u;
  uint32_t dim = 0u;
  std::span<float> output;
};

struct execute_norm_rows {
  const norm_rows_request &request;
  dispatch_result &result;
};

struct execute_unit_rows {
  const unit_rows_request &request;
  dispatch_result &result;
};

} // namespace emel::kernel::zcrms::event
