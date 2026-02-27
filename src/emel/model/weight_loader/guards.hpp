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

struct apply_has_effect_errors {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    if (!apply_count_matches{}(ev, ctx)) {
      return false;
    }
    for (const auto & result : ev.request.results) {
      if (result.err != emel::error::cast(error::none)) {
        return true;
      }
    }
    return false;
  }
};

struct valid_apply {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return has_bound_tensors{}(ctx) && apply_count_matches{}(ev, ctx) &&
           !apply_has_effect_errors{}(ev, ctx);
  }
};

struct invalid_apply_request {
  bool operator()(const event::apply_runtime & ev, const action::context & ctx) const noexcept {
    return !has_bound_tensors{}(ctx) || !apply_count_matches{}(ev, ctx);
  }
};

}  // namespace emel::model::weight_loader::guard
