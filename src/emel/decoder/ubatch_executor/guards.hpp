#pragma once

#include "emel/decoder/ubatch_executor/actions.hpp"

namespace emel::decoder::ubatch_executor::guard {

inline constexpr auto phase_ok = [](const action::context & ctx) {
  return ctx.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & ctx) {
  return ctx.phase_error != EMEL_OK;
};

inline constexpr auto outputs_produced_invalid = [](const action::context & ctx) {
  return ctx.phase_error == EMEL_OK && ctx.outputs_produced <= 0;
};

inline constexpr auto always = [](const action::context &) {
  return true;
};

struct valid_execute_request {
  bool operator()(const event::execute & ev, const action::context &) const noexcept {
    if (ev.memory_coordinator_sm == nullptr || ev.kv_cache_sm == nullptr) {
      return false;
    }
    if (ev.ubatch_index < 0 || ev.ubatch_size <= 0) {
      return false;
    }
    return true;
  }
};

struct invalid_execute_request {
  bool operator()(const event::execute & ev, const action::context & ctx) const noexcept {
    return !valid_execute_request{}(ev, ctx);
  }
};

}  // namespace emel::decoder::ubatch_executor::guard
