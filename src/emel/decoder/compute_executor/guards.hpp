#pragma once

#include "emel/decoder/compute_executor/actions.hpp"

namespace emel::decoder::compute_executor::guard {

inline constexpr auto graph_reused = [](const action::context & ctx) {
  return ctx.graph_reused;
};

inline constexpr auto graph_needs_allocation = [](const action::context & ctx) {
  return !ctx.graph_reused;
};

inline constexpr auto phase_ok = [](const action::context & ctx) {
  return ctx.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & ctx) {
  return ctx.phase_error != EMEL_OK;
};

inline constexpr auto always = [](const action::context &) {
  return true;
};

struct valid_execute_request {
  bool operator()(const event::execute & ev, const action::context &) const noexcept {
    if (ev.prepare_graph == nullptr ||
        ev.bind_inputs == nullptr ||
        ev.run_backend == nullptr ||
        ev.extract_outputs == nullptr) {
      return false;
    }
    if (ev.ubatch_index < 0 || ev.ubatch_size <= 0) {
      return false;
    }
    if (ev.kv_tokens < 0) {
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

}  // namespace emel::decoder::compute_executor::guard
