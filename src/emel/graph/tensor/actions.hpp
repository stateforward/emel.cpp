#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/graph/tensor/context.hpp"
#include "emel/graph/tensor/detail.hpp"
#include "emel/graph/tensor/errors.hpp"
#include "emel/graph/tensor/events.hpp"

namespace emel::graph::tensor::action {

struct begin_reserve_tensor {
  void operator()(const detail::reserve_tensor_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_publish_filled_tensor {
  void operator()(const detail::publish_filled_tensor_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_release_tensor_ref {
  void operator()(const detail::release_tensor_ref_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_reset_tensor_epoch {
  void operator()(const detail::reset_tensor_epoch_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_capture_tensor_state {
  void operator()(const detail::capture_tensor_state_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct exec_reserve_tensor {
  void operator()(const detail::reserve_tensor_runtime & ev, context & ctx) const noexcept {
    ev.ctx.accepted = ctx.tensors.process_indexed<tensor::detail::op_reserve>(
        static_cast<size_t>(ev.request.tensor_id),
        tensor::detail::op_reserve{
          .buffer = ev.request.buffer,
          .buffer_bytes = ev.request.buffer_bytes,
          .consumer_refs = static_cast<uint32_t>(ev.request.consumer_refs),
          .is_leaf = static_cast<uint8_t>(ev.request.is_leaf),
        });
  }
};

struct exec_publish_filled_tensor {
  void operator()(const detail::publish_filled_tensor_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ev.ctx.accepted = ctx.tensors.process_indexed<tensor::detail::op_publish_filled>(tensor_id);
  }
};

struct exec_release_tensor_ref {
  void operator()(const detail::release_tensor_ref_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ev.ctx.accepted = ctx.tensors.process_indexed<tensor::detail::op_release_ref>(tensor_id);
  }
};

struct exec_reset_tensor_epoch {
  void operator()(const detail::reset_tensor_epoch_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ev.ctx.accepted = ctx.tensors.process_indexed<tensor::detail::op_reset_epoch>(tensor_id);
  }
};

struct exec_capture_tensor_state {
  void operator()(const detail::capture_tensor_state_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    const auto & storage = ctx.tensors.storage();
    event::tensor_state & out = *ev.request.state_out;

    out.lifecycle_state = static_cast<event::lifecycle>(storage.lifecycle[tensor_id]);
    out.is_leaf = storage.kind[tensor_id];
    out.seed_refs = storage.seed_refs[tensor_id];
    out.live_refs = storage.live_refs[tensor_id];
    out.buffer = storage.buffer[tensor_id];
    out.buffer_bytes = storage.buffer_bytes[tensor_id];

    ev.ctx.accepted = true;
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    runtime_ev.ctx.ok = false;
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct mark_internal_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::internal_error);
    runtime_ev.ctx.ok = false;
    runtime_ev.error_code_out = static_cast<int32_t>(runtime_ev.ctx.err);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = tensor::detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.ctx.ok = true;
    runtime_ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    auto & runtime_ev = tensor::detail::unwrap_runtime_event(ev);
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

inline constexpr begin_reserve_tensor begin_reserve_tensor{};
inline constexpr begin_publish_filled_tensor begin_publish_filled_tensor{};
inline constexpr begin_release_tensor_ref begin_release_tensor_ref{};
inline constexpr begin_reset_tensor_epoch begin_reset_tensor_epoch{};
inline constexpr begin_capture_tensor_state begin_capture_tensor_state{};
inline constexpr exec_reserve_tensor exec_reserve_tensor{};
inline constexpr exec_publish_filled_tensor exec_publish_filled_tensor{};
inline constexpr exec_release_tensor_ref exec_release_tensor_ref{};
inline constexpr exec_reset_tensor_epoch exec_reset_tensor_epoch{};
inline constexpr exec_capture_tensor_state exec_capture_tensor_state{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::tensor::action
