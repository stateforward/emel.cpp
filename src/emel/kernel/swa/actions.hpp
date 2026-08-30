#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/kernel/swa/events.hpp"

namespace emel::kernel::swa::action {

// The swa kernel holds no persistent actor state.
struct context {};

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
  uint32_t i = 0u;
  for (; i + 8u <= dim; i += 8u)
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i), _mm256_loadu_ps(rhs + i),
                          sum);
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, sum);
  float out = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] +
              lanes[6] + lanes[7];
  for (; i < dim; ++i)
    out += lhs[i] * rhs[i];
  return out;
#else
  float out = 0.0f;
  for (uint32_t i = 0u; i < dim; ++i)
    out += lhs[i] * rhs[i];
  return out;
#endif
}

EMEL_KERNEL_SWA_AVX2_TARGET inline void
accumulate_value_avx2(float *output, const float *values, const float weight,
                      const uint32_t dim) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const __m256 weight_v = _mm256_set1_ps(weight);
  uint32_t i = 0u;
  for (; i + 8u <= dim; i += 8u)
    _mm256_storeu_ps(output + i,
                     _mm256_fmadd_ps(weight_v, _mm256_loadu_ps(values + i),
                                     _mm256_loadu_ps(output + i)));
  for (; i < dim; ++i)
    output[i] += weight * values[i];
#else
  for (uint32_t i = 0u; i < dim; ++i)
    output[i] += weight * values[i];
#endif
}

struct effect_execute_attend {
  EMEL_KERNEL_SWA_AVX2_TARGET void operator()(const event::execute_attend &ev,
                                              context &) const noexcept {
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
              dot_avx2(query_row, key, request.head_dim) * inv_scale;
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
          accumulate_value_avx2(output_row, value,
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
