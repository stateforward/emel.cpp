#pragma once

#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <cstdint>

#include "emel/kernel/zcrms/detail.hpp"
#include "emel/kernel/zcrms/events.hpp"

namespace emel::kernel::zcrms::action {

// The zcrms kernel holds no persistent actor state.
struct context {};

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_ZCRMS_ACTION_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define EMEL_KERNEL_ZCRMS_ACTION_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_ZCRMS_ACTION_AVX2_TARGET
#endif

struct effect_execute_norm_rows {
  EMEL_KERNEL_ZCRMS_ACTION_AVX2_TARGET void
  operator()(const event::execute_norm_rows &ev, context &) const noexcept {
    const auto &request = ev.request;
    for (uint32_t row = 0u; row < request.rows; ++row) {
      const size_t base = static_cast<size_t>(row) * request.dim;
      const float inv_rms =
          detail::compute_inv_rms(request.input.data() + base, request.dim);
      uint32_t i = 0u;
#if defined(__x86_64__) || defined(_M_X64)
      const __m256 one = _mm256_set1_ps(1.0f);
      const __m256 inv = _mm256_set1_ps(inv_rms);
      for (; i + 8u <= request.dim; i += 8u) {
        const __m256 scale =
            _mm256_add_ps(one, _mm256_loadu_ps(request.scale.data() + i));
        const __m256 input = _mm256_loadu_ps(request.input.data() + base + i);
        _mm256_storeu_ps(request.output.data() + base + i,
                         _mm256_mul_ps(_mm256_mul_ps(scale, input), inv));
      }
#endif
      for (; i < request.dim; ++i)
        request.output[base + i] =
            (1.0f + request.scale[i]) * request.input[base + i] * inv_rms;
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_unit_rows {
  EMEL_KERNEL_ZCRMS_ACTION_AVX2_TARGET void
  operator()(const event::execute_unit_rows &ev, context &) const noexcept {
    const auto &request = ev.request;
    for (uint32_t row = 0u; row < request.rows; ++row) {
      const size_t base = static_cast<size_t>(row) * request.dim;
      const float inv_rms =
          detail::compute_inv_rms(request.input.data() + base, request.dim);
      uint32_t i = 0u;
#if defined(__x86_64__) || defined(_M_X64)
      const __m256 inv = _mm256_set1_ps(inv_rms);
      for (; i + 8u <= request.dim; i += 8u)
        _mm256_storeu_ps(
            request.output.data() + base + i,
            _mm256_mul_ps(_mm256_loadu_ps(request.input.data() + base + i),
                          inv));
#endif
      for (; i < request.dim; ++i)
        request.output[base + i] = request.input[base + i] * inv_rms;
    }
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::zcrms::action
