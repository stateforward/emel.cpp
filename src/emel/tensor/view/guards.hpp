#pragma once

#include "emel/tensor/view/context.hpp"
#include "emel/tensor/view/detail.hpp"
#include "emel/tensor/view/errors.hpp"

namespace emel::tensor::view::guard {

struct capture_tensor_view_request_valid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev, const action::context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    using policy = typename std::remove_cvref_t<decltype(runtime_ev)>::policy_type;
    return runtime_ev.request.tensor_machine != nullptr &&
           runtime_ev.request.state_out != nullptr &&
           runtime_ev.request.tensor_id >= 0 &&
           runtime_ev.request.tensor_id < policy::max_tensors;
  }
};

struct capture_tensor_view_request_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev, const action::context & ctx) const noexcept {
    return !capture_tensor_view_request_valid{}(ev, ctx);
  }
};

struct operation_succeeded {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.accepted &&
           runtime_ev.ctx.err == emel::error::cast(error::none);
  }
};

struct operation_failed_with_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err != emel::error::cast(error::none);
  }
};

struct operation_failed_without_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.accepted &&
           runtime_ev.ctx.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::tensor::view::guard
