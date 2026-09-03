#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/kernel/detail.hpp"
#include "emel/kernel/hadamard/events.hpp"

namespace emel::kernel::hadamard::action {

// The hadamard kernel holds no persistent actor state.
struct context {};

namespace detail {

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_HADAMARD_AVX2_TARGET                                       \
  __attribute__((target("avx2,fma,f16c")))
#else
#define EMEL_KERNEL_HADAMARD_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_HADAMARD_AVX2_TARGET
#endif
inline uint16_t load_fp16(const uint8_t *bytes) noexcept {
  return static_cast<uint16_t>(bytes[0]) | static_cast<uint16_t>(bytes[1])
                                               << 8u;
}

EMEL_KERNEL_HADAMARD_AVX2_TARGET inline void
fwht512_avx2(float *values) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) &&                                \
    ((defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||           \
     defined(__GNUC__) || defined(__clang__))
  const __m256 sign1 =
      _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
  const __m256 sign2 =
      _mm256_setr_ps(1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f);
  const __m256 sign4 =
      _mm256_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
  for (uint32_t base = 0u; base < 512u; base += 8u) {
    const __m256 x = _mm256_loadu_ps(values + base);
    const __m256 swapped1 = _mm256_permute_ps(x, 0xb1);
    const __m256 sums1 = _mm256_add_ps(x, swapped1);
    const __m256 diffs1 = _mm256_mul_ps(_mm256_sub_ps(x, swapped1), sign1);
    const __m256 stage1 = _mm256_blend_ps(sums1, diffs1, 0xaau);
    const __m256 swapped2 = _mm256_permute_ps(stage1, 0x4e);
    const __m256 sums2 = _mm256_add_ps(stage1, swapped2);
    const __m256 diffs2 = _mm256_mul_ps(_mm256_sub_ps(stage1, swapped2), sign2);
    const __m256 stage2 = _mm256_blend_ps(sums2, diffs2, 0xccu);
    const __m256 swapped4 = _mm256_permute2f128_ps(stage2, stage2, 0x01);
    const __m256 sums4 = _mm256_add_ps(stage2, swapped4);
    const __m256 diffs4 = _mm256_mul_ps(_mm256_sub_ps(stage2, swapped4), sign4);
    _mm256_storeu_ps(values + base, _mm256_blend_ps(sums4, diffs4, 0xf0u));
  }
  for (uint32_t step = 8u; step < 512u; step <<= 1u)
    for (uint32_t base = 0u; base < 512u; base += step << 1u)
      for (uint32_t j = 0u; j < step; j += 8u) {
        const __m256 a = _mm256_loadu_ps(values + base + j);
        const __m256 b = _mm256_loadu_ps(values + base + step + j);
        _mm256_storeu_ps(values + base + j, _mm256_add_ps(a, b));
        _mm256_storeu_ps(values + base + step + j, _mm256_sub_ps(a, b));
      }
  const __m256 scale = _mm256_set1_ps(0.044194173824159220275f);
  for (uint32_t i = 0u; i < 512u; i += 8u)
    _mm256_storeu_ps(values + i,
                     _mm256_mul_ps(_mm256_loadu_ps(values + i), scale));
#else
  emel::kernel::detail::fwht_normalized(values, 512u);
#endif
}

EMEL_KERNEL_HADAMARD_AVX2_TARGET inline void
silu512_avx2(const float *input, const uint8_t *d2, float *output) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) &&                                \
    ((defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||           \
     defined(__GNUC__) || defined(__clang__))
  alignas(32) float silu_lanes[8];
  for (uint32_t i = 0u; i < 512u; i += 8u) {
    const __m256 diagonal = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(d2 + i * 2u)));
    const __m256 values = _mm256_mul_ps(diagonal, _mm256_loadu_ps(input + i));
    _mm256_store_ps(silu_lanes, values);
    for (float &value : silu_lanes)
      value = value / (1.0f + std::exp(-value));
    _mm256_storeu_ps(output + i, _mm256_load_ps(silu_lanes));
  }
#else
  namespace quant = emel::kernel::detail::quant;
  for (uint32_t i = 0u; i < 512u; ++i) {
    const float value =
        emel::kernel::detail::quant::fp16_to_fp32(load_fp16(d2 + i * 2u)) *
        input[i];
    output[i] = value / (1.0f + std::exp(-value));
  }
#endif
}
EMEL_KERNEL_HADAMARD_AVX2_TARGET inline void
execute_mlp_row_avx2(const event::mlp_row_request &request) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) &&                                \
    ((defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||           \
     defined(__GNUC__) || defined(__clang__))
  const auto d1 = request.d1.data();
  const auto d3 = request.d3.data();
  float *lane = request.workspace.data();
  for (uint32_t i = 0u; i < 512u; i += 8u) {
    const __m256 input = _mm256_loadu_ps(request.input.data() + i);
    const __m256 diagonal = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(d1 + i * 2u)));
    _mm256_storeu_ps(lane + i, _mm256_mul_ps(diagonal, input));
  }
  fwht512_avx2(lane);
  silu512_avx2(lane, request.d2.data(), lane);
  fwht512_avx2(lane);
  for (uint32_t i = 0u; i < 512u; i += 8u) {
    const __m256 skip = _mm256_loadu_ps(request.skip.data() + i);
    const __m256 diagonal = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(d3 + i * 2u)));
    const __m256 output =
        _mm256_fmadd_ps(diagonal, _mm256_loadu_ps(lane + i), skip);
    _mm256_storeu_ps(request.output.data() + i, output);
  }
#else
  (void)request;
#endif
}

} // namespace detail

struct effect_execute_mlp_row {
  void operator()(const event::execute_mlp_row &ev, context &) const noexcept {
    namespace quant = emel::kernel::detail::quant;
    const auto &request = ev.request;
    const uint32_t n = request.hada_n;
    float *lane = request.workspace.data();
    const uint8_t *d1 = request.d1.data();
    const uint8_t *d2 = request.d2.data();
    const uint8_t *d3 = request.d3.data();
    // Padding beyond d_model is zero (hada_n padding semantics); the split
    // loops keep the data plane branch-free.
    for (uint32_t i = 0u; i < request.d_model; ++i)
      lane[i] = quant::fp16_to_fp32(detail::load_fp16(d1 + i * 2u)) *
                request.input[i];
    for (uint32_t i = request.d_model; i < n; ++i)
      lane[i] = 0.0f;
    emel::kernel::detail::fwht_normalized(lane, n);
    for (uint32_t i = 0u; i < n; ++i) {
      const float value =
          quant::fp16_to_fp32(detail::load_fp16(d2 + i * 2u)) * lane[i];
      lane[i] = value / (1.0f + std::exp(-value));
    }
    emel::kernel::detail::fwht_normalized(lane, n);
    for (uint32_t i = 0u; i < request.d_model; ++i)
      request.output[i] =
          request.skip[i] +
          quant::fp16_to_fp32(detail::load_fp16(d3 + i * 2u)) * lane[i];
    ev.result.accepted = true;
  }
};

struct effect_execute_mlp_row_avx2 {
  void operator()(const event::execute_mlp_row_avx2 &ev,
                  context &) const noexcept {
    detail::execute_mlp_row_avx2(ev.request);
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::hadamard::action
