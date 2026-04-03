#pragma once

#include "emel/model/tensor/context.hpp"
#include "emel/model/tensor/detail.hpp"
#include "emel/model/tensor/errors.hpp"

namespace emel::model::tensor::guard {

namespace detail {

inline bool valid_tensor_id(const int32_t tensor_id) noexcept {
  return tensor_id >= 0 && tensor_id < tensor::detail::max_tensors;
}

template <class runtime_event_type>
bool operation_dispatched(const runtime_event_type & ev) noexcept {
  return tensor::detail::unwrap_runtime_event(ev).ctx.accepted;
}

}  // namespace detail

struct bind_tensor_request_valid {
  bool operator()(const tensor::detail::bind_tensor_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ev.request.buffer != nullptr &&
           ev.request.buffer_bytes > 0u &&
           ev.request.tensor_record.data_size > 0u;
  }
};

struct bind_tensor_request_invalid {
  bool operator()(const tensor::detail::bind_tensor_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !bind_tensor_request_valid{}(ev, ctx);
  }
};

struct evict_tensor_request_valid {
  bool operator()(const tensor::detail::evict_tensor_runtime & ev,
                  const action::context & ctx) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ctx.tensors.lifecycle[static_cast<size_t>(ev.request.tensor_id)] !=
               event::lifecycle::unbound;
  }
};

struct evict_tensor_request_invalid {
  bool operator()(const tensor::detail::evict_tensor_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !evict_tensor_request_valid{}(ev, ctx);
  }
};

struct capture_tensor_state_request_valid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) && ev.request.state_out != nullptr;
  }
};

struct capture_tensor_state_request_invalid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !capture_tensor_state_request_valid{}(ev, ctx);
  }
};

struct operation_succeeded {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return detail::operation_dispatched(ev) &&
           tensor::detail::unwrap_runtime_event(ev).ctx.err ==
               emel::error::cast(error::none);
  }
};

struct operation_not_dispatched {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !detail::operation_dispatched(ev);
  }
};

}  // namespace emel::model::tensor::guard
