#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/kernel/attention/context.hpp"
#include "emel/kernel/attention/events.hpp"
#include "emel/kernel/detail.hpp"

namespace emel::kernel::attention::action {

struct effect_execute_head_range {
  void operator()(const event::execute &ev, context &ctx) const noexcept {
    ev.result = {};
    const auto &request = ev.request;
    const float scale = 1.0f / std::sqrt(static_cast<float>(request.head_dim));

    for (int32_t head = request.head_begin; head < request.head_end; ++head) {
      const int32_t head_offset = head * request.head_dim;
      const float *q_head =
          request.query.data() + static_cast<std::size_t>(head_offset);
      for (int32_t dim = 0; dim < request.head_dim; ++dim) {
        ctx.q_bf16[static_cast<std::size_t>(dim)] =
            emel::kernel::detail::fp32_to_bf16(q_head[dim]);
      }
      for (int32_t physical = 0; physical < request.position_capacity;
           ++physical) {
        ctx.scores[static_cast<std::size_t>(physical)] =
            -std::numeric_limits<float>::infinity();
      }
      for (int32_t position = 0; position < request.valid_positions;
           ++position) {
        const int64_t unwrapped =
            static_cast<int64_t>(request.physical_begin) + position;
        const int32_t physical = static_cast<int32_t>(
            unwrapped -
            static_cast<int64_t>(unwrapped >= request.position_capacity) *
                request.position_capacity);
        const std::size_t cache_begin =
            request.layer_offset +
            static_cast<std::size_t>(physical) *
                static_cast<std::size_t>(request.hidden_dim) +
            static_cast<std::size_t>(head_offset);
        ctx.scores[static_cast<std::size_t>(physical)] =
            emel::kernel::detail::vec_dot_bf16_ggml(
                request.head_dim, request.key_cache.data() + cache_begin,
                ctx.q_bf16.data()) *
            scale;
      }

      emel::kernel::detail::soft_max_row_ggml(request.position_capacity,
                                              ctx.scores.data());
      for (int32_t physical = 0; physical < request.position_capacity;
           ++physical) {
        ctx.weights_bf16[static_cast<std::size_t>(physical)] =
            emel::kernel::detail::fp32_to_bf16(
                ctx.scores[static_cast<std::size_t>(physical)]);
      }

      float *attention_head =
          request.output.data() +
          static_cast<std::size_t>(head - request.head_begin) *
              static_cast<std::size_t>(request.head_dim);
      std::fill_n(ctx.output_accumulators.data(), request.head_dim, 0.0);
      for (int32_t position = 0; position < request.valid_positions;
           ++position) {
        const int64_t unwrapped =
            static_cast<int64_t>(request.physical_begin) + position;
        const int32_t physical = static_cast<int32_t>(
            unwrapped -
            static_cast<int64_t>(unwrapped >= request.position_capacity) *
                request.position_capacity);
        const float weight = emel::kernel::detail::bf16_to_fp32(
            ctx.weights_bf16[static_cast<std::size_t>(physical)]);
        const uint16_t *value_head =
            request.value_cache.data() + request.layer_offset +
            static_cast<std::size_t>(physical) *
                static_cast<std::size_t>(request.hidden_dim) +
            static_cast<std::size_t>(head_offset);
        for (int32_t dim = 0; dim < request.head_dim; ++dim) {
          ctx.output_accumulators[static_cast<std::size_t>(dim)] +=
              static_cast<double>(
                  emel::kernel::detail::bf16_to_fp32(value_head[dim]) * weight);
        }
      }
      for (int32_t dim = 0; dim < request.head_dim; ++dim) {
        attention_head[dim] = static_cast<float>(
            ctx.output_accumulators[static_cast<std::size_t>(dim)]);
      }
    }
    ev.result.accepted = true;
  }
};

struct effect_reject_execute {
  void operator()(const event::execute &ev, context &) const noexcept {
    ev.result.accepted = false;
  }
};

struct effect_accept_execute {
  void operator()(const event::execute &, context &) const noexcept {}
};

struct effect_emit_execute_done {
  void operator()(const event::execute &ev, context &) const noexcept {
    ev.on_done(events::execute_done{.request = &ev});
  }
};

struct effect_emit_execute_error {
  void operator()(const event::execute &ev, context &) const noexcept {
    ev.on_error(events::execute_error{.request = &ev});
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.result.accepted; }) {
      ev.result.accepted = false;
    }
  }
};

} // namespace emel::kernel::attention::action
