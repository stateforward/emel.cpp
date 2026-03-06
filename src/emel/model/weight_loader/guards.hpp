#pragma once

#include "emel/model/weight_loader/context.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

struct valid_bind {
  bool operator()(const event::bind_runtime & ev) const noexcept {
    return ev.request.tensors.data() != nullptr && !ev.request.tensors.empty();
  }
};

struct invalid_bind {
  bool operator()(const event::bind_runtime & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct has_bound_tensors {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.tensors != nullptr && ctx.tensor_count > 0u;
  }
};

struct has_sufficient_effect_capacity {
  bool operator()(const event::plan_runtime & ev, const action::context & ctx) const noexcept {
    return ev.request.effects.size() >= static_cast<std::size_t>(ctx.tensor_count);
  }
};

struct valid_plan {
  bool operator()(const event::plan_runtime & ev, const action::context & ctx) const noexcept {
    return has_bound_tensors{}(ctx) && has_sufficient_effect_capacity{}(ev, ctx);
  }
};

struct invalid_plan_request {
  bool operator()(const event::plan_runtime &, const action::context & ctx) const noexcept {
    return !has_bound_tensors{}(ctx);
  }
};

struct invalid_plan_capacity {
  bool operator()(const event::plan_runtime & ev, const action::context & ctx) const noexcept {
    return has_bound_tensors{}(ctx) && !has_sufficient_effect_capacity{}(ev, ctx);
  }
};

struct apply_count_matches {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return ev.request.results.size() == static_cast<std::size_t>(ctx.planned_effects);
  }
};

struct valid_apply_request {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return has_bound_tensors{}(ctx) && apply_count_matches{}(ev, ctx);
  }
};

struct invalid_apply_request {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_apply_request{}(ev, ctx);
  }
};

struct apply_effect_errors_present {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.has_effect_errors;
  }
};

struct apply_effect_errors_absent {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return !ev.ctx.has_effect_errors;
  }
};

template <class runtime_event_type>
inline emel::error::type runtime_error(const runtime_event_type & ev) noexcept {
  return ev.ctx.err;
}

template <class runtime_event_type>
inline bool error_is(const runtime_event_type & ev,
                     const emel::error::type expected) noexcept {
  return runtime_error(ev) == expected;
}

template <class runtime_event_type>
inline bool error_is_unknown(const runtime_event_type & ev) noexcept {
  return !error_is(ev, emel::error::cast(error::none)) &&
         !error_is(ev, emel::error::cast(error::invalid_request)) &&
         !error_is(ev, emel::error::cast(error::capacity)) &&
         !error_is(ev, emel::error::cast(error::backend_error)) &&
         !error_is(ev, emel::error::cast(error::model_invalid)) &&
         !error_is(ev, emel::error::cast(error::out_of_memory)) &&
         !error_is(ev, emel::error::cast(error::internal_error)) &&
         !error_is(ev, emel::error::cast(error::untracked));
}

struct bind_error_none {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct bind_error_invalid_request {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct bind_error_capacity {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::capacity));
  }
};

struct bind_error_backend_error {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct bind_error_model_invalid {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct bind_error_out_of_memory {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::out_of_memory));
  }
};

struct bind_error_internal_error {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct bind_error_untracked {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct bind_error_unknown {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return error_is_unknown(ev);
  }
};

struct plan_error_none {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct plan_error_invalid_request {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct plan_error_capacity {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::capacity));
  }
};

struct plan_error_backend_error {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct plan_error_model_invalid {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct plan_error_out_of_memory {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::out_of_memory));
  }
};

struct plan_error_internal_error {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct plan_error_untracked {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct plan_error_unknown {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return error_is_unknown(ev);
  }
};

struct apply_error_none {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::none));
  }
};

struct apply_error_invalid_request {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct apply_error_capacity {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::capacity));
  }
};

struct apply_error_backend_error {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct apply_error_model_invalid {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::model_invalid));
  }
};

struct apply_error_out_of_memory {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::out_of_memory));
  }
};

struct apply_error_internal_error {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct apply_error_untracked {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is(ev, emel::error::cast(error::untracked));
  }
};

struct apply_error_unknown {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return error_is_unknown(ev);
  }
};

struct bind_done_callback_present {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct bind_done_callback_absent {
  bool operator()(const event::bind_runtime & ev, const action::context & ctx) const noexcept {
    return !bind_done_callback_present{}(ev, ctx);
  }
};

struct bind_error_callback_present {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct bind_error_callback_absent {
  bool operator()(const event::bind_runtime & ev, const action::context & ctx) const noexcept {
    return !bind_error_callback_present{}(ev, ctx);
  }
};

struct plan_done_callback_present {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct plan_done_callback_absent {
  bool operator()(const event::plan_runtime & ev, const action::context & ctx) const noexcept {
    return !plan_done_callback_present{}(ev, ctx);
  }
};

struct plan_error_callback_present {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct plan_error_callback_absent {
  bool operator()(const event::plan_runtime & ev, const action::context & ctx) const noexcept {
    return !plan_error_callback_present{}(ev, ctx);
  }
};

struct apply_done_callback_present {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct apply_done_callback_absent {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return !apply_done_callback_present{}(ev, ctx);
  }
};

struct apply_error_callback_present {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct apply_error_callback_absent {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return !apply_error_callback_present{}(ev, ctx);
  }
};

}  // namespace emel::model::weight_loader::guard
