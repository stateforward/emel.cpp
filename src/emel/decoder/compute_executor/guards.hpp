#pragma once

#include "emel/decoder/compute_executor/actions.hpp"

namespace emel::decoder::compute_executor::guard {

inline constexpr auto graph_reused = [](const events::prepare_graph_done & ev) {
  return ev.reused;
};

inline constexpr auto graph_needs_allocation = [](const events::prepare_graph_done & ev) {
  return !ev.reused;
};

struct valid_execute_request {
  bool operator()(const event::validate & ev, const action::context &) const noexcept {
    const event::execute * request = ev.request;
    if (request == nullptr) {
      return false;
    }
    if (request->prepare_graph == nullptr ||
        request->bind_inputs == nullptr ||
        request->run_backend == nullptr ||
        request->extract_outputs == nullptr) {
      return false;
    }
    if (request->ubatch_index < 0 || request->ubatch_size <= 0) {
      return false;
    }
    if (request->kv_tokens < 0) {
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

struct valid_prepare_graph_request {
  bool operator()(const event::prepare_graph & ev, const action::context &) const noexcept {
    const event::execute * request = ev.request;
    if (request == nullptr) {
      return false;
    }
    if (request->prepare_graph == nullptr) {
      return false;
    }
    if (ev.reused_out == nullptr) {
      return false;
    }
    return true;
  }
};

struct invalid_prepare_graph_request {
  bool operator()(const event::prepare_graph & ev, const action::context & ctx) const noexcept {
    return !valid_prepare_graph_request{}(ev, ctx);
  }
};

struct valid_alloc_graph_request {
  bool operator()(const event::alloc_graph & ev, const action::context &) const noexcept {
    const event::execute * request = ev.request;
    return request != nullptr && request->alloc_graph != nullptr;
  }
};

struct invalid_alloc_graph_request {
  bool operator()(const event::alloc_graph & ev, const action::context & ctx) const noexcept {
    return !valid_alloc_graph_request{}(ev, ctx);
  }
};

struct valid_bind_inputs_request {
  bool operator()(const event::bind_inputs & ev, const action::context &) const noexcept {
    const event::execute * request = ev.request;
    return request != nullptr && request->bind_inputs != nullptr;
  }
};

struct invalid_bind_inputs_request {
  bool operator()(const event::bind_inputs & ev, const action::context & ctx) const noexcept {
    return !valid_bind_inputs_request{}(ev, ctx);
  }
};

struct valid_run_backend_request {
  bool operator()(const event::run_backend & ev, const action::context &) const noexcept {
    const event::execute * request = ev.request;
    return request != nullptr && request->run_backend != nullptr;
  }
};

struct invalid_run_backend_request {
  bool operator()(const event::run_backend & ev, const action::context & ctx) const noexcept {
    return !valid_run_backend_request{}(ev, ctx);
  }
};

struct valid_extract_outputs_request {
  bool operator()(const event::extract_outputs & ev, const action::context &) const noexcept {
    const event::execute * request = ev.request;
    return request != nullptr && request->extract_outputs != nullptr;
  }
};

struct invalid_extract_outputs_request {
  bool operator()(const event::extract_outputs & ev, const action::context & ctx) const noexcept {
    return !valid_extract_outputs_request{}(ev, ctx);
  }
};

}  // namespace emel::decoder::compute_executor::guard
