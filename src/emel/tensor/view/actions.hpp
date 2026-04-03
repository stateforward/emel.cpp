#pragma once

#include <array>
#include <cstdint>

#include "emel/tensor/view/context.hpp"
#include "emel/tensor/view/detail.hpp"
#include "emel/tensor/view/errors.hpp"

namespace emel::tensor::view::action {

struct begin_capture_tensor_view {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = false;
    runtime_ev.ctx.accepted = false;
    runtime_ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct exec_capture_tensor_view {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    using policy = typename std::remove_cvref_t<decltype(runtime_ev)>::policy_type;
    int32_t tensor_error = policy::none_error_code;
    runtime_ev.ctx.accepted = policy::capture_tensor_state(*runtime_ev.request.tensor_machine,
                                                           runtime_ev.request.tensor_id,
                                                           *runtime_ev.request.state_out,
                                                           tensor_error);
    const std::array<emel::error::type, 3> mapped_errors = {
      static_cast<emel::error::type>(tensor_error),
      emel::error::cast(error::invalid_request),
      emel::error::cast(error::internal_error),
    };
    const size_t from_invalid_request =
        static_cast<size_t>(tensor_error == policy::invalid_request_error_code);
    const size_t from_internal_error =
        static_cast<size_t>(tensor_error == policy::internal_error_code);
    const size_t mapped_index = from_invalid_request + (from_internal_error * 2u);
    runtime_ev.ctx.err = mapped_errors[mapped_index];
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_internal_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::internal_error);
    runtime_ev.ctx.ok = false;
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_error_from_operation {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.ok = false;
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
    runtime_ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.ok = false;
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; ev.error_code_out; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      ev.ctx.ok = false;
      ev.error_code_out = static_cast<int32_t>(ev.ctx.err);
    }
  }
};

inline constexpr begin_capture_tensor_view begin_capture_tensor_view{};
inline constexpr exec_capture_tensor_view exec_capture_tensor_view{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr mark_error_from_operation mark_error_from_operation{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::tensor::view::action
