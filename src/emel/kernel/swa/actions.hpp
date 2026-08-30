#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/swa/events.hpp"

namespace emel::kernel::swa::action {

// The swa kernel holds no persistent actor state.
struct context {};

struct effect_execute_attend {
  void operator()(const event::execute_attend &ev, context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t span_len = request.position - request.window_begin + 1u;
    const uint32_t reps = request.heads / request.kv_heads;
    const float inv_scale =
        1.0f / std::sqrt(static_cast<float>(request.head_dim));
    const size_t head_stride =
        static_cast<size_t>(request.capacity) * request.head_dim;
    for (uint32_t head = 0u; head < request.heads; ++head) {
      const uint32_t kv_head = head / reps;
      const float *query_row =
          request.query.data() + static_cast<size_t>(head) * request.head_dim;
      const float *key_base = request.key_cache.data() + kv_head * head_stride;
      const float *value_base =
          request.value_cache.data() + kv_head * head_stride;
      float max_score = -3.402823466e+38f;
      for (uint32_t offset = 0u; offset < span_len; ++offset) {
        const uint32_t logical = request.window_begin + offset;
        const size_t slot =
            static_cast<size_t>(logical % request.capacity) * request.head_dim;
        float dot = 0.0f;
        for (uint32_t i = 0u; i < request.head_dim; ++i)
          dot += query_row[i] * key_base[slot + i];
        const float score = dot * inv_scale;
        request.workspace[offset] = score;
        max_score = std::max(max_score, score);
      }
      float weight_sum = 0.0f;
      for (uint32_t offset = 0u; offset < span_len; ++offset) {
        const float weight = std::exp(request.workspace[offset] - max_score);
        request.workspace[offset] = weight;
        weight_sum += weight;
      }
      const float inv_sum = 1.0f / weight_sum;
      float *output_row =
          request.output.data() + static_cast<size_t>(head) * request.head_dim;
      for (uint32_t i = 0u; i < request.head_dim; ++i)
        output_row[i] = 0.0f;
      for (uint32_t offset = 0u; offset < span_len; ++offset) {
        const uint32_t logical = request.window_begin + offset;
        const size_t slot =
            static_cast<size_t>(logical % request.capacity) * request.head_dim;
        const float weight = request.workspace[offset] * inv_sum;
        for (uint32_t i = 0u; i < request.head_dim; ++i)
          output_row[i] += weight * value_base[slot + i];
      }
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
