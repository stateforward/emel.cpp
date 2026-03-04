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

struct bind_phase_ok {
  bool operator()(const event::bind_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct bind_phase_failed {
  bool operator()(const event::bind_runtime & ev, const action::context & ctx) const noexcept {
    return !bind_phase_ok{}(ev, ctx);
  }
};

struct plan_phase_ok {
  bool operator()(const event::plan_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct plan_phase_failed {
  bool operator()(const event::plan_runtime & ev, const action::context & ctx) const noexcept {
    return !plan_phase_ok{}(ev, ctx);
  }
};

struct apply_phase_ok {
  bool operator()(const event::apply_runtime & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct apply_phase_failed {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return !apply_phase_ok{}(ev, ctx);
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
