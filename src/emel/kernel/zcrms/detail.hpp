#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace emel::kernel::zcrms::detail {

inline constexpr float k_epsilon = 1e-6f;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_ZCRMS_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define EMEL_KERNEL_ZCRMS_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_ZCRMS_AVX2_TARGET
#endif

// 1 / sqrt(mean(x^2) + eps): the shared RMS denominator used by the
// ZCRMSNorm and unit-RMS ops and by the engram alpha gate.
EMEL_KERNEL_ZCRMS_AVX2_TARGET inline float
compute_inv_rms(const float *values, const uint32_t dim) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  __m256 sum = _mm256_setzero_ps();
  uint32_t i = 0u;
  for (; i + 8u <= dim; i += 8u) {
    const __m256 value = _mm256_loadu_ps(values + i);
    sum = _mm256_fmadd_ps(value, value, sum);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, sum);
  float sum_squares = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                      lanes[5] + lanes[6] + lanes[7];
  for (; i < dim; ++i)
    sum_squares += values[i] * values[i];
#else
  float sum_squares = 0.0f;
  for (uint32_t i = 0u; i < dim; ++i)
    sum_squares += values[i] * values[i];
#endif
  return 1.0f / std::sqrt(sum_squares / static_cast<float>(dim) + k_epsilon);
}

} // namespace emel::kernel::zcrms::detail
