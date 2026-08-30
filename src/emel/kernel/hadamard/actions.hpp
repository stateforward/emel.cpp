#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/hadamard/events.hpp"

namespace emel::kernel::hadamard::action {

// The hadamard kernel holds no persistent actor state.
struct context {};

struct effect_execute_mlp_row {
  void operator()(const event::execute_mlp_row &ev, context &) const noexcept {
    namespace quant = emel::kernel::detail::quant;
    const auto &request = ev.request;
    const uint32_t n = request.hada_n;
    float *lane = request.workspace.data();
    const uint16_t *d1 = reinterpret_cast<const uint16_t *>(request.d1.data());
    const uint16_t *d2 = reinterpret_cast<const uint16_t *>(request.d2.data());
    const uint16_t *d3 = reinterpret_cast<const uint16_t *>(request.d3.data());
    // Padding beyond d_model is zero (hada_n padding semantics); the split
    // loops keep the data plane branch-free.
    for (uint32_t i = 0u; i < request.d_model; ++i)
      lane[i] = quant::fp16_to_fp32(d1[i]) * request.input[i];
    for (uint32_t i = request.d_model; i < n; ++i)
      lane[i] = 0.0f;
    emel::kernel::detail::fwht_normalized(lane, n);
    for (uint32_t i = 0u; i < n; ++i) {
      const float value = quant::fp16_to_fp32(d2[i]) * lane[i];
      lane[i] = value / (1.0f + std::exp(-value));
    }
    emel::kernel::detail::fwht_normalized(lane, n);
    for (uint32_t i = 0u; i < request.d_model; ++i)
      request.output[i] =
          request.skip[i] + quant::fp16_to_fp32(d3[i]) * lane[i];
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::hadamard::action
