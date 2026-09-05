#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/kernel/swa/detail.hpp"
#include "emel/kernel/swa/events.hpp"
#include "emel/kernel/x86_64/context.hpp"

namespace emel::kernel::swa::action {

// CPU capability is detected once with machine construction, never in dispatch.
struct context {
  bool avx2_fma_available = emel::kernel::x86_64::detail::detect_avx2() &&
                            emel::kernel::x86_64::detail::detect_fma();
};

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_SWA_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define EMEL_KERNEL_SWA_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_SWA_AVX2_TARGET
#endif

EMEL_KERNEL_SWA_AVX2_TARGET inline float
dot_avx2(const float *lhs, const float *rhs, const uint32_t dim) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  __m256 sum = _mm256_setzero_ps();
  size_t i = 0u;
  const size_t vector_end = static_cast<size_t>(dim) & ~size_t{7u};
  for (; i < vector_end; i += 8u)
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i), _mm256_loadu_ps(rhs + i),
                          sum);
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, sum);
  float out = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] +
              lanes[6] + lanes[7];
  for (; i < static_cast<size_t>(dim); ++i)
    out += lhs[i] * rhs[i];
  return out;
#else
  float out = 0.0f;
  for (size_t i = 0u; i < static_cast<size_t>(dim); ++i)
    out += lhs[i] * rhs[i];
  return out;
#endif
}

EMEL_KERNEL_SWA_AVX2_TARGET inline void
dot_pair_avx2(const float *lhs0, const float *lhs1, const float *rhs,
              const uint32_t dim, float &out0, float &out1) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  size_t i = 0u;
  const size_t vector_end = static_cast<size_t>(dim) & ~size_t{7u};
  for (; i < vector_end; i += 8u) {
    const __m256 rhs_v = _mm256_loadu_ps(rhs + i);
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs0 + i), rhs_v, sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs1 + i), rhs_v, sum1);
  }
  alignas(32) float lanes0[8];
  alignas(32) float lanes1[8];
  _mm256_store_ps(lanes0, sum0);
  _mm256_store_ps(lanes1, sum1);
  out0 = lanes0[0] + lanes0[1] + lanes0[2] + lanes0[3] + lanes0[4] + lanes0[5] +
         lanes0[6] + lanes0[7];
  out1 = lanes1[0] + lanes1[1] + lanes1[2] + lanes1[3] + lanes1[4] + lanes1[5] +
         lanes1[6] + lanes1[7];
  for (; i < static_cast<size_t>(dim); ++i) {
    out0 += lhs0[i] * rhs[i];
    out1 += lhs1[i] * rhs[i];
  }
#else
  out0 = 0.0f;
  out1 = 0.0f;
  for (size_t i = 0u; i < static_cast<size_t>(dim); ++i) {
    out0 += lhs0[i] * rhs[i];
    out1 += lhs1[i] * rhs[i];
  }
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline void
accumulate_value_pair_scalar(float *output0, float *output1,
                             const float *values, const float weight0,
                             const float weight1, const uint32_t dim) noexcept {
  for (size_t i = 0u; i < static_cast<size_t>(dim); ++i) {
    output0[i] += weight0 * values[i];
    output1[i] += weight1 * values[i];
  }
}

EMEL_KERNEL_SWA_AVX2_TARGET inline void
accumulate_value_avx2(float *output, const float *values, const float weight,
                      const uint32_t dim) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const __m256 weight_v = _mm256_set1_ps(weight);
  size_t i = 0u;
  const size_t vector_end = static_cast<size_t>(dim) & ~size_t{7u};
  for (; i < vector_end; i += 8u)
    _mm256_storeu_ps(output + i,
                     _mm256_fmadd_ps(weight_v, _mm256_loadu_ps(values + i),
                                     _mm256_loadu_ps(output + i)));
  for (; i < static_cast<size_t>(dim); ++i)
    output[i] += weight * values[i];
#else
  for (size_t i = 0u; i < static_cast<size_t>(dim); ++i)
    output[i] += weight * values[i];
#endif
}

inline float dot_scalar(const float *lhs, const float *rhs,
                        const uint32_t dim) noexcept {
  float sum = 0.0f;
  for (size_t i = 0u; i < static_cast<size_t>(dim); ++i)
    sum += lhs[i] * rhs[i];
  return sum;
}

inline void accumulate_value_scalar(float *output, const float *values,
                                    const float weight,
                                    const uint32_t dim) noexcept {
  for (size_t i = 0u; i < static_cast<size_t>(dim); ++i)
    output[i] += weight * values[i];
}

struct effect_execute_attend {
  void operator()(const event::execute_attend &ev, context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t span_len = request.position - request.window_begin + 1u;
    const uint32_t reps = request.heads / request.kv_heads;
    const float inv_scale =
        1.0f / std::sqrt(static_cast<float>(request.head_dim));
    const size_t head_stride =
        static_cast<size_t>(request.capacity) * request.head_dim;
    const uint32_t first_slot = request.window_begin % request.capacity;
    const uint32_t first_len =
        std::min(span_len, request.capacity - first_slot);
    const uint32_t second_len = span_len - first_len;
    for (uint32_t head = 0u; head < request.heads; ++head) {
      const uint32_t kv_head = head / reps;
      const float *query_row =
          request.query.data() + static_cast<size_t>(head) * request.head_dim;
      const float *key_base = request.key_cache.data() + kv_head * head_stride;
      const float *value_base =
          request.value_cache.data() + kv_head * head_stride;
      float max_score = -3.402823466e+38f;
      uint32_t offset = 0u;
      const auto score_segment = [&](const uint32_t slot_begin,
                                     const uint32_t count) {
        const float *key =
            key_base + static_cast<size_t>(slot_begin) * request.head_dim;
        for (uint32_t row = 0u; row < count; ++row, ++offset) {
          const float score =
              dot_scalar(query_row, key, request.head_dim) * inv_scale;
          request.workspace[offset] = score;
          max_score = std::max(max_score, score);
          key += request.head_dim;
        }
      };
      score_segment(first_slot, first_len);
      score_segment(0u, second_len);
      float weight_sum = 0.0f;
      for (offset = 0u; offset < span_len; ++offset) {
        const float weight = std::exp(request.workspace[offset] - max_score);
        request.workspace[offset] = weight;
        weight_sum += weight;
      }
      const float inv_sum = 1.0f / weight_sum;
      float *output_row =
          request.output.data() + static_cast<size_t>(head) * request.head_dim;
      std::fill_n(output_row, request.head_dim, 0.0f);
      offset = 0u;
      const auto value_segment = [&](const uint32_t slot_begin,
                                     const uint32_t count) {
        const float *value =
            value_base + static_cast<size_t>(slot_begin) * request.head_dim;
        for (uint32_t row = 0u; row < count; ++row, ++offset) {
          accumulate_value_scalar(output_row, value,
                                  request.workspace[offset] * inv_sum,
                                  request.head_dim);
          value += request.head_dim;
        }
      };
      value_segment(first_slot, first_len);
      value_segment(0u, second_len);
    }
    ev.result.accepted = true;
  }
};

template <bool vector_exp> struct effect_execute_attend_gqa2_avx2_impl {
  EMEL_KERNEL_SWA_AVX2_TARGET void operator()(
      const std::conditional_t<vector_exp,
                               event::execute_attend_gqa2_avx2_vector_exp,
                               event::execute_attend_gqa2_avx2> &ev,
      context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t span_len = request.position - request.window_begin + 1u;
    const float inv_scale =
        1.0f / std::sqrt(static_cast<float>(request.head_dim));
    const size_t head_stride =
        static_cast<size_t>(request.capacity) * request.head_dim;
    const uint32_t first_slot = request.window_begin % request.capacity;
    const uint32_t first_len =
        std::min(span_len, request.capacity - first_slot);
    const uint32_t second_len = span_len - first_len;

    for (uint32_t kv_head = 0u; kv_head < request.kv_heads; ++kv_head) {
      const uint32_t head0 = kv_head * 2u;
      const uint32_t head1 = head0 + 1u;
      const float *query0 =
          request.query.data() + static_cast<size_t>(head0) * request.head_dim;
      const float *query1 =
          request.query.data() + static_cast<size_t>(head1) * request.head_dim;
      const float *key_base =
          request.key_cache.data() + static_cast<size_t>(kv_head) * head_stride;
      const float *value_base = request.value_cache.data() +
                                static_cast<size_t>(kv_head) * head_stride;
      float *score0 = request.workspace.data();
      float *score1 = request.workspace.data() + span_len;
      float max0 = -3.402823466e+38f;
      float max1 = -3.402823466e+38f;
      uint32_t offset = 0u;
      const auto score_segment = [&](const uint32_t slot_begin,
                                     const uint32_t count) {
        const float *key =
            key_base + static_cast<size_t>(slot_begin) * request.head_dim;
        for (uint32_t row = 0u; row < count; ++row, ++offset) {
          float dot0 = 0.0f;
          float dot1 = 0.0f;
          dot_pair_avx2(query0, query1, key, request.head_dim, dot0, dot1);
          const float value0 = dot0 * inv_scale;
          const float value1 = dot1 * inv_scale;
          score0[offset] = value0;
          score1[offset] = value1;
          max0 = std::max(max0, value0);
          max1 = std::max(max1, value1);
          key += request.head_dim;
        }
      };
      score_segment(first_slot, first_len);
      score_segment(0u, second_len);

      float sum0 = 0.0f;
      float sum1 = 0.0f;
      if constexpr (vector_exp) {
        sum0 = detail::exp_sum_avx2(score0, span_len, max0);
        sum1 = detail::exp_sum_avx2(score1, span_len, max1);
      } else {
        for (offset = 0u; offset < span_len; ++offset) {
          const float weight = std::exp(score0[offset] - max0);
          score0[offset] = weight;
          sum0 += weight;
        }
        for (offset = 0u; offset < span_len; ++offset) {
          const float weight = std::exp(score1[offset] - max1);
          score1[offset] = weight;
          sum1 += weight;
        }
      }

      const float inv_sum0 = 1.0f / sum0;
      const float inv_sum1 = 1.0f / sum1;
      float *output0 =
          request.output.data() + static_cast<size_t>(head0) * request.head_dim;
      float *output1 =
          request.output.data() + static_cast<size_t>(head1) * request.head_dim;
      std::fill_n(output0, request.head_dim, 0.0f);
      std::fill_n(output1, request.head_dim, 0.0f);
      offset = 0u;
      const auto value_segment = [&](const uint32_t slot_begin,
                                     const uint32_t count) {
        const float *value =
            value_base + static_cast<size_t>(slot_begin) * request.head_dim;
        for (uint32_t row = 0u; row < count; ++row, ++offset) {
          accumulate_value_pair_scalar(
              output0, output1, value, score0[offset] * inv_sum0,
              score1[offset] * inv_sum1, request.head_dim);
          value += request.head_dim;
        }
      };
      value_segment(first_slot, first_len);
      value_segment(0u, second_len);
    }
    ev.result.accepted = true;
  }
};

using effect_execute_attend_gqa2_avx2 =
    effect_execute_attend_gqa2_avx2_impl<false>;
using effect_execute_attend_gqa2_avx2_vector_exp =
    effect_execute_attend_gqa2_avx2_impl<true>;
struct effect_execute_cache_write {
  void operator()(const event::execute_cache_write &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const size_t head_stride =
        static_cast<size_t>(request.capacity) * request.head_dim;
    const size_t slot =
        static_cast<size_t>(request.position % request.capacity) *
        request.head_dim;
    for (uint32_t head = 0u; head < request.kv_heads; ++head) {
      const size_t src = static_cast<size_t>(head) * request.head_dim;
      const size_t dst = head * head_stride + slot;
      for (uint32_t i = 0u; i < request.head_dim; ++i) {
        request.key_cache[dst + i] = request.key_rows[src + i];
        request.value_cache[dst + i] = request.value_rows[src + i];
      }
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_gate_mul {
  void operator()(const event::execute_gate_mul &ev, context &) const noexcept {
    const auto &request = ev.request;
    for (uint32_t i = 0u; i < request.dim; ++i)
      request.values[i] *= 1.0f / (1.0f + std::exp(-request.gate_logits[i]));
    ev.result.accepted = true;
  }
};

struct effect_execute_residual_gate {
  void operator()(const event::execute_residual_gate &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const float gate = 1.0f / (1.0f + std::exp(-request.gate));
    for (uint32_t i = 0u; i < request.dim; ++i)
      request.output[i] = request.skip[i] + gate * request.values[i];
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::swa::action
