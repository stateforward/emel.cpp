#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/rope/events.hpp"

namespace emel::kernel::rope::action {

// The rope kernel holds no persistent actor state.
struct context {};

struct effect_execute_precompute {
  void operator()(const event::execute_precompute &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t half = request.head_dim / 2u;
    for (uint32_t position = 0u; position < request.positions; ++position) {
      const size_t base = static_cast<size_t>(position) * half;
      for (uint32_t i = 0u; i < half; ++i) {
        const float freq =
            1.0f /
            std::pow(request.theta, static_cast<float>(2u * i) /
                                        static_cast<float>(request.head_dim));
        const float angle = static_cast<float>(position) * freq;
        request.cos_out[base + i] = std::cos(angle);
        request.sin_out[base + i] = std::sin(angle);
      }
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_apply_rows {
  void operator()(const event::execute_apply_rows &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t half = request.head_dim / 2u;
    const size_t table_base = static_cast<size_t>(request.position) * half;
    for (uint32_t head = 0u; head < request.head_count; ++head) {
      float *row =
          request.rows.data() + static_cast<size_t>(head) * request.head_dim;
      for (uint32_t i = 0u; i < half; ++i) {
        const float cos_value = request.cos_table[table_base + i];
        const float sin_value = request.sin_table[table_base + i];
        const float x1 = row[i];
        const float x2 = row[half + i];
        row[i] = x1 * cos_value - x2 * sin_value;
        row[half + i] = x2 * cos_value + x1 * sin_value;
      }
    }
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::rope::action
