#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/engram/events.hpp"
#include "emel/kernel/zcrms/detail.hpp"

namespace emel::kernel::engram::action {

// The engram kernel holds no persistent actor state.
struct context {};

inline constexpr uint32_t k_hash_seed = 0x9E3779B9u;
inline constexpr uint32_t k_hash_prime = 0x01000193u;

struct effect_execute_hash_rows {
  void operator()(const event::execute_hash_rows &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t tables = request.num_orders * request.heads;
    for (uint32_t position = 0u; position < request.positions; ++position) {
      for (uint32_t order_index = 0u; order_index < request.num_orders;
           ++order_index) {
        const uint32_t order = request.orders[order_index];
        for (uint32_t head = 0u; head < request.heads; ++head) {
          const uint32_t table = order_index * request.heads + head;
          uint32_t acc = k_hash_seed * (table + 1u);
          for (uint32_t j = 0u; j < order; ++j) {
            // Positions before the window read as token 0 (reference zero
            // padding); branch-free via an in-range multiplier mask.
            const uint32_t in_range = static_cast<uint32_t>(j <= position);
            const uint32_t source = position - j * in_range;
            const uint32_t token =
                in_range * static_cast<uint32_t>(request.tokens[source]);
            acc = (acc ^ token) * k_hash_prime;
          }
          acc ^= acc >> 15u;
          const size_t slot = static_cast<size_t>(position) * tables + table;
          request.indices[slot] = acc % request.slots;
          const uint32_t span_ok =
              static_cast<uint32_t>(position + 1u >= order);
          const uint32_t oldest = position - (order - 1u) * span_ok;
          request.ngram_ok[slot] = static_cast<float>(
              span_ok * static_cast<uint32_t>(request.valid[oldest] != 0u));
        }
      }
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_conv_taps {
  void operator()(const event::execute_conv_taps &ev,
                  context &) const noexcept {
    namespace quant = emel::kernel::detail::quant;
    const auto &request = ev.request;
    const uint16_t *taps =
        reinterpret_cast<const uint16_t *>(request.taps.data());
    for (uint32_t i = 0u; i < request.dim; ++i)
      request.output[i] = 0.0f;
    for (uint32_t tap = 0u; tap < request.conv_taps; ++tap) {
      const float tap_ok = static_cast<float>(request.tap_valid[tap] != 0u);
      const float *value_row =
          request.value_rows.data() + static_cast<size_t>(tap) * request.dim;
      const uint16_t *tap_row = taps + static_cast<size_t>(tap) * request.dim;
      for (uint32_t i = 0u; i < request.dim; ++i)
        request.output[i] +=
            tap_ok * quant::fp16_to_fp32(tap_row[i]) * value_row[i];
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_alpha_gate {
  void operator()(const event::execute_alpha_gate &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const float inv_rms_u = emel::kernel::zcrms::detail::compute_inv_rms(
        request.u.data(), request.dim);
    const float inv_rms_key = emel::kernel::zcrms::detail::compute_inv_rms(
        request.key.data(), request.dim);
    float dot = 0.0f;
    for (uint32_t i = 0u; i < request.dim; ++i)
      dot += request.u[i] * inv_rms_u * request.key[i] * inv_rms_key;
    const float scaled = dot / std::sqrt(static_cast<float>(request.dim));
    const float alpha = 1.0f / (1.0f + std::exp(-scaled));
    for (uint32_t i = 0u; i < request.dim; ++i)
      request.output[i] = request.u[i] + alpha * request.value[i];
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::engram::action
