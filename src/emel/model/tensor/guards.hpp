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
bool operation_dispatched(const runtime_event_type &ev) noexcept {
  return tensor::detail::unwrap_runtime_event(ev).ctx.accepted;
}

} // namespace detail

struct storage_bind_valid {
  bool operator()(const event::bind_storage &ev) const noexcept {
    return ev.tensors.data() != nullptr && !ev.tensors.empty() &&
           ev.tensors.size() <=
               static_cast<size_t>(tensor::detail::max_tensors);
  }
};

struct storage_bind_invalid {
  bool operator()(const event::bind_storage &ev) const noexcept {
    return !storage_bind_valid{}(ev);
  }
};

struct storage_bound {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.bound_records.data() != nullptr && !ctx.bound_records.empty();
  }
};

struct plan_load_has_capacity {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.effects.size() >= ctx.bound_records.size();
  }
};

struct plan_load_valid {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bound{}(ctx) && plan_load_has_capacity{}(ev, ctx);
  }
};

struct plan_load_invalid_request {
  template <class event_type>
  bool operator()(const event_type &,
                  const action::context &ctx) const noexcept {
    return !storage_bound{}(ctx);
  }
};

struct plan_load_invalid_capacity {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bound{}(ctx) && !plan_load_has_capacity{}(ev, ctx);
  }
};

struct apply_results_count_matches {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    return request.results.size() == ctx.bound_records.size();
  }
};

struct apply_results_valid {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return storage_bound{}(ctx) && apply_results_count_matches{}(ev, ctx);
  }
};

struct apply_results_invalid {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_results_valid{}(ev, ctx);
  }
};

struct apply_effect_errors_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    const auto &request = tensor::detail::request_event(ev);
    uint32_t error_flags = 0u;
    for (const auto &result : request.results) {
      error_flags |=
          static_cast<uint32_t>(result.err != emel::error::cast(error::none));
    }
    return error_flags != 0u;
  }
};

struct apply_effect_errors_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_effect_errors_present{}(ev, ctx);
  }
};

struct apply_results_valid_with_effect_errors {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return apply_results_valid{}(ev, ctx) &&
           apply_effect_errors_present{}(ev, ctx);
  }
};

struct apply_results_valid_without_effect_errors {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return apply_results_valid{}(ev, ctx) &&
           apply_effect_errors_absent{}(ev, ctx);
  }
};

template <class runtime_event_type>
inline emel::error::type runtime_error(const runtime_event_type &ev) noexcept {
  return tensor::detail::unwrap_runtime_event(ev).ctx.err;
}

template <class runtime_event_type>
inline bool error_is(const runtime_event_type &ev,
                     const emel::error::type expected) noexcept {
  return runtime_error(ev) == expected;
}

template <class runtime_event_type>
inline bool error_is_unknown(const runtime_event_type &ev) noexcept {
  return !error_is(ev, emel::error::cast(error::none)) &&
         !error_is(ev, emel::error::cast(error::invalid_request)) &&
         !error_is(ev, emel::error::cast(error::capacity)) &&
         !error_is(ev, emel::error::cast(error::backend_error)) &&
         !error_is(ev, emel::error::cast(error::model_invalid)) &&
         !error_is(ev, emel::error::cast(error::out_of_memory)) &&
         !error_is(ev, emel::error::cast(error::internal_error)) &&
         !error_is(ev, emel::error::cast(error::untracked));
}

struct error_none {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct error_invalid_request {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct error_capacity {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::capacity));
  }
};

struct error_backend_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct error_model_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct error_out_of_memory {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::out_of_memory));
  }
};

struct error_internal_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct error_untracked {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct error_unknown {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev,
                  const action::context &) const noexcept {
    return error_is_unknown(ev);
  }
};

struct bind_storage_done_callback_present {
  bool operator()(const event::bind_storage &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_done);
  }
};

struct bind_storage_done_callback_absent {
  bool operator()(const event::bind_storage &ev,
                  const action::context &ctx) const noexcept {
    return !bind_storage_done_callback_present{}(ev, ctx);
  }
};

struct bind_storage_error_callback_present {
  bool operator()(const event::bind_storage &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_error);
  }
};

struct bind_storage_error_callback_absent {
  bool operator()(const event::bind_storage &ev,
                  const action::context &ctx) const noexcept {
    return !bind_storage_error_callback_present{}(ev, ctx);
  }
};

struct plan_load_done_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_done);
  }
};

struct plan_load_done_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !plan_load_done_callback_present{}(ev, ctx);
  }
};

struct plan_load_error_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_error);
  }
};

struct plan_load_error_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !plan_load_error_callback_present{}(ev, ctx);
  }
};

struct apply_effect_results_done_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_done);
  }
};

struct apply_effect_results_done_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_effect_results_done_callback_present{}(ev, ctx);
  }
};

struct apply_effect_results_error_callback_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(tensor::detail::request_event(ev).on_error);
  }
};

struct apply_effect_results_error_callback_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !apply_effect_results_error_callback_present{}(ev, ctx);
  }
};

struct error_code_output_present {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return tensor::detail::unwrap_runtime_event(ev).error_code_out != nullptr;
  }
};

struct error_code_output_absent {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return !error_code_output_present{}(ev);
  }
};

struct bind_tensor_request_valid {
  bool operator()(const tensor::detail::bind_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ev.request.buffer != nullptr && ev.request.buffer_bytes > 0u &&
           ev.request.tensor_record.data_size > 0u;
  }
};

struct bind_tensor_request_invalid {
  bool operator()(const tensor::detail::bind_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !bind_tensor_request_valid{}(ev, ctx);
  }
};

struct evict_tensor_request_valid {
  bool operator()(const tensor::detail::evict_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ctx.tensors.lifecycle[static_cast<size_t>(ev.request.tensor_id)] !=
               event::lifecycle::unbound;
  }
};

struct evict_tensor_request_invalid {
  bool operator()(const tensor::detail::evict_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !evict_tensor_request_valid{}(ev, ctx);
  }
};

struct capture_tensor_state_request_valid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime &ev,
                  const action::context &) const noexcept {
    return detail::valid_tensor_id(ev.request.tensor_id) &&
           ev.request.state_out != nullptr;
  }
};

struct capture_tensor_state_request_invalid {
  bool operator()(const tensor::detail::capture_tensor_state_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !capture_tensor_state_request_valid{}(ev, ctx);
  }
};

struct operation_succeeded {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return detail::operation_dispatched(ev) &&
           tensor::detail::unwrap_runtime_event(ev).ctx.err ==
               emel::error::cast(error::none);
  }
};

struct operation_not_dispatched {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &ev) const noexcept {
    return !detail::operation_dispatched(ev);
  }
};

} // namespace emel::model::tensor::guard
