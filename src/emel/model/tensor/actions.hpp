#pragma once

#include "emel/model/tensor/context.hpp"
#include "emel/model/tensor/detail.hpp"
#include "emel/model/tensor/errors.hpp"

namespace emel::model::tensor::action {

struct begin_bind_tensor {
  void operator()(const detail::bind_tensor_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.ok = false;
    ev.ctx.accepted = false;
    ev.error_code_out = static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct begin_evict_tensor {
  void operator()(const detail::evict_tensor_runtime & ev, context &) const noexcept {
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

struct exec_bind_tensor {
  void operator()(const detail::bind_tensor_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ctx.tensors.lifecycle[tensor_id] = event::lifecycle::resident;
    ctx.tensors.buffer[tensor_id] = ev.request.buffer;
    ctx.tensors.buffer_bytes[tensor_id] = ev.request.buffer_bytes;
    ctx.tensors.file_offset[tensor_id] = ev.request.tensor_record.file_offset;
    ctx.tensors.data_size[tensor_id] = ev.request.tensor_record.data_size;
    ctx.tensors.file_index[tensor_id] = ev.request.tensor_record.file_index;
    ctx.tensors.tensor_type[tensor_id] = ev.request.tensor_record.type;
    ev.ctx.accepted = true;
  }
};

struct exec_evict_tensor {
  void operator()(const detail::evict_tensor_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    ctx.tensors.lifecycle[tensor_id] = event::lifecycle::evicted;
    ctx.tensors.buffer[tensor_id] = nullptr;
    ctx.tensors.buffer_bytes[tensor_id] = 0u;
    ev.ctx.accepted = true;
  }
};

struct exec_capture_tensor_state {
  void operator()(const detail::capture_tensor_state_runtime & ev, context & ctx) const noexcept {
    const size_t tensor_id = static_cast<size_t>(ev.request.tensor_id);
    event::tensor_state & out = *ev.request.state_out;
    out.lifecycle_state = ctx.tensors.lifecycle[tensor_id];
    out.buffer = ctx.tensors.buffer[tensor_id];
    out.buffer_bytes = ctx.tensors.buffer_bytes[tensor_id];
    out.file_offset = ctx.tensors.file_offset[tensor_id];
    out.data_size = ctx.tensors.data_size[tensor_id];
    out.file_index = ctx.tensors.file_index[tensor_id];
    out.tensor_type = ctx.tensors.tensor_type[tensor_id];
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

inline constexpr begin_bind_tensor begin_bind_tensor{};
inline constexpr begin_evict_tensor begin_evict_tensor{};
inline constexpr begin_capture_tensor_state begin_capture_tensor_state{};
inline constexpr exec_bind_tensor exec_bind_tensor{};
inline constexpr exec_evict_tensor exec_evict_tensor{};
inline constexpr exec_capture_tensor_state exec_capture_tensor_state{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::model::tensor::action
