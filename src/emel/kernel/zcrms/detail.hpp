#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace emel::kernel::zcrms::detail {

inline constexpr float k_epsilon = 1e-6f;

// 1 / sqrt(mean(x^2) + eps): the shared RMS denominator used by the
// ZCRMSNorm and unit-RMS ops and by the engram alpha gate.
inline float compute_inv_rms(const float *values, const uint32_t dim) noexcept {
  float sum_squares = 0.0f;
  for (uint32_t i = 0u; i < dim; ++i)
    sum_squares += values[i] * values[i];
  return 1.0f / std::sqrt(sum_squares / static_cast<float>(dim) + k_epsilon);
}

} // namespace emel::kernel::zcrms::detail
