#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/kernel/zcrms/detail.hpp"
#include "emel/kernel/zcrms/events.hpp"

namespace emel::kernel::zcrms::action {

// The zcrms kernel holds no persistent actor state.
struct context {};

struct effect_execute_norm_rows {
  void operator()(const event::execute_norm_rows &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    for (uint32_t row = 0u; row < request.rows; ++row) {
      const size_t base = static_cast<size_t>(row) * request.dim;
      const float inv_rms =
          detail::compute_inv_rms(request.input.data() + base, request.dim);
      for (uint32_t i = 0u; i < request.dim; ++i)
        request.output[base + i] =
            (1.0f + request.scale[i]) * request.input[base + i] * inv_rms;
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_unit_rows {
  void operator()(const event::execute_unit_rows &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    for (uint32_t row = 0u; row < request.rows; ++row) {
      const size_t base = static_cast<size_t>(row) * request.dim;
      const float inv_rms =
          detail::compute_inv_rms(request.input.data() + base, request.dim);
      for (uint32_t i = 0u; i < request.dim; ++i)
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
