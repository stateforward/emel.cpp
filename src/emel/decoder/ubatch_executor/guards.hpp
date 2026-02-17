#pragma once

#include "emel/decoder/ubatch_executor/actions.hpp"

namespace emel::decoder::ubatch_executor::guard {

inline constexpr auto rollback_required = [](const action::context & ctx) {
  (void)ctx;
  return true;
};

struct valid_execute_request {
  bool operator()(const event::validate & ev, const action::context & ctx) const noexcept {
    const event::execute * request = ev.request;
    if (request == nullptr) {
      return false;
    }
    if (request->memory_coordinator_sm == nullptr || request->kv_cache_sm == nullptr) {
      return false;
    }
    if (ctx.ubatch_index < 0 || ctx.ubatch_size <= 0) {
      return false;
    }
    return true;
  }
};

struct invalid_execute_request {
  bool operator()(const event::validate & ev, const action::context & ctx) const noexcept {
    return !valid_execute_request{}(ev, ctx);
  }
};

struct valid_prepare_memory_request {
  bool operator()(const event::prepare_memory & ev, const action::context &) const noexcept {
    return ev.memory_coordinator_sm != nullptr;
  }
};

struct invalid_prepare_memory_request {
  bool operator()(const event::prepare_memory & ev, const action::context & ctx) const noexcept {
    return !valid_prepare_memory_request{}(ev, ctx);
  }
};

struct valid_prepare_kv_request {
  bool operator()(const event::prepare_kv & ev, const action::context &) const noexcept {
    return ev.kv_cache_sm != nullptr;
  }
};

struct invalid_prepare_kv_request {
  bool operator()(const event::prepare_kv & ev, const action::context & ctx) const noexcept {
    return !valid_prepare_kv_request{}(ev, ctx);
  }
};

struct valid_run_compute_request {
  bool operator()(const event::run_compute & ev, const action::context &) const noexcept {
    return ev.kv_cache_sm != nullptr && ev.request != nullptr;
  }
};

struct invalid_run_compute_request {
  bool operator()(const event::run_compute & ev, const action::context & ctx) const noexcept {
    return !valid_run_compute_request{}(ev, ctx);
  }
};

struct valid_extract_outputs_request {
  bool operator()(const event::extract_outputs &, const action::context & ctx) const noexcept {
    return ctx.outputs_produced > 0;
  }
};

struct invalid_extract_outputs_request {
  bool operator()(const event::extract_outputs & ev, const action::context & ctx) const noexcept {
    return !valid_extract_outputs_request{}(ev, ctx);
  }
};

struct valid_rollback_request {
  bool operator()(const event::rollback & ev, const action::context &) const noexcept {
    return ev.kv_cache_sm != nullptr;
  }
};

struct invalid_rollback_request {
  bool operator()(const event::rollback & ev, const action::context & ctx) const noexcept {
    return !valid_rollback_request{}(ev, ctx);
  }
};

}  // namespace emel::decoder::ubatch_executor::guard
