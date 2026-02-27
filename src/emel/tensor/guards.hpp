#pragma once

#include <cstddef>

#include "emel/tensor/context.hpp"
#include "emel/tensor/detail.hpp"
#include "emel/tensor/errors.hpp"
#include "emel/tensor/events.hpp"

namespace emel::tensor::guard {

namespace detail {

inline bool valid_tensor_id(const int32_t tensor_id) noexcept {
  return tensor_id >= 0 && tensor_id < tensor::detail::max_tensors;
}

}  // namespace detail

struct reserve_tensor_request_valid {
  bool operator()(const tensor::detail::reserve_tensor_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ev.request.buffer != nullptr &&
           ev.request.buffer_bytes > 0u &&
           ev.request.consumer_refs >= 0;
  }
};

struct reserve_tensor_request_invalid {
  bool operator()(const tensor::detail::reserve_tensor_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !reserve_tensor_request_valid{}(ev, ctx);
  }
};

struct publish_filled_tensor_request_valid {
  bool operator()(const tensor::detail::publish_filled_tensor_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id);
  }
};

struct publish_filled_tensor_request_invalid {
  bool operator()(const tensor::detail::publish_filled_tensor_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !publish_filled_tensor_request_valid{}(ev, ctx);
  }
};

struct release_tensor_ref_request_valid {
  bool operator()(const tensor::detail::release_tensor_ref_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id);
  }
};

struct release_tensor_ref_request_invalid {
  bool operator()(const tensor::detail::release_tensor_ref_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !release_tensor_ref_request_valid{}(ev, ctx);
  }
};

struct reset_tensor_epoch_request_valid {
  bool operator()(const tensor::detail::reset_tensor_epoch_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id);
  }
};

struct reset_tensor_epoch_request_invalid {
  bool operator()(const tensor::detail::reset_tensor_epoch_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !reset_tensor_epoch_request_valid{}(ev, ctx);
  }
};

struct capture_tensor_state_request_valid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime & ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ev.request.state_out != nullptr;
  }
};

struct capture_tensor_state_request_invalid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime & ev,
                  const action::context & ctx) const noexcept {
    return !capture_tensor_state_request_valid{}(ev, ctx);
  }
};

namespace detail {

template <class runtime_event_type>
bool operation_dispatched(const runtime_event_type & ev) noexcept {
  return tensor::detail::unwrap_runtime_event(ev).ctx.accepted;
}

template <class runtime_event_type>
bool lifecycle_is_internal_error(const runtime_event_type & ev, const action::context & ctx) noexcept {
  const auto & runtime_ev = tensor::detail::unwrap_runtime_event(ev);
  const size_t tensor_id = static_cast<size_t>(runtime_ev.request.tensor_id);
  return ctx.tensors.storage().lifecycle[tensor_id] == tensor::detail::lifecycle_state::internal_error;
}

}  // namespace detail

struct operation_succeeded {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev, const action::context & ctx) const noexcept {
    return detail::operation_dispatched(ev) &&
           !detail::lifecycle_is_internal_error(ev, ctx);
  }
};

struct operation_failed_internal {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev, const action::context & ctx) const noexcept {
    return detail::operation_dispatched(ev) &&
           detail::lifecycle_is_internal_error(ev, ctx);
  }
};

struct operation_not_dispatched {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !detail::operation_dispatched(ev);
  }
};

struct capture_operation_succeeded {
  bool operator()(const tensor::detail::capture_tensor_state_runtime & ev,
                  const action::context &) const noexcept {
    return detail::operation_dispatched(ev);
  }
};

struct capture_operation_not_dispatched {
  bool operator()(const tensor::detail::capture_tensor_state_runtime & ev,
                  const action::context &) const noexcept {
    return !detail::operation_dispatched(ev);
  }
};

}  // namespace emel::tensor::guard
