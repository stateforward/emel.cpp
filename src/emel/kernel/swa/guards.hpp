#pragma once

#include "emel/kernel/swa/actions.hpp"

namespace emel::kernel::swa::guard {

struct guard_execute_attend {
  bool operator()(const event::execute_attend &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    if (request.heads == 0u || request.kv_heads == 0u ||
        request.head_dim == 0u || request.capacity == 0u ||
        (request.heads % request.kv_heads) != 0u ||
        request.window_begin > request.position)
      return false;
    const uint32_t span_len = request.position - request.window_begin + 1u;
    const uint64_t cache = static_cast<uint64_t>(request.kv_heads) *
                           request.capacity * request.head_dim;
    const uint64_t q_len =
        static_cast<uint64_t>(request.heads) * request.head_dim;
    return span_len <= request.capacity &&
           request.workspace.size() >= span_len &&
           request.query.size() >= q_len && request.output.size() >= q_len &&
           request.key_cache.size() >= cache &&
           request.value_cache.size() >= cache;
  }
};

struct guard_execute_cache_write {
  bool operator()(const event::execute_cache_write &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t rows =
        static_cast<uint64_t>(request.kv_heads) * request.head_dim;
    const uint64_t cache = static_cast<uint64_t>(request.kv_heads) *
                           request.capacity * request.head_dim;
    return request.kv_heads > 0u && request.head_dim > 0u &&
           request.capacity > 0u && request.key_rows.size() >= rows &&
           request.value_rows.size() >= rows &&
           request.key_cache.size() >= cache &&
           request.value_cache.size() >= cache;
  }
};

struct guard_execute_gate_mul {
  bool operator()(const event::execute_gate_mul &ev,
                  const action::context &) const noexcept {
    return ev.request.dim > 0u && ev.request.values.size() >= ev.request.dim &&
           ev.request.gate_logits.size() >= ev.request.dim;
  }
};

struct guard_execute_residual_gate {
  bool operator()(const event::execute_residual_gate &ev,
                  const action::context &) const noexcept {
    return ev.request.dim > 0u && ev.request.skip.size() >= ev.request.dim &&
           ev.request.values.size() >= ev.request.dim &&
           ev.request.output.size() >= ev.request.dim;
  }
};

} // namespace emel::kernel::swa::guard
