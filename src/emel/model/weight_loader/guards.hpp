#pragma once

#include "emel/model/weight_loader/context.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.last_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.last_error != EMEL_OK;
  }
};

struct valid_bind {
  bool operator()(const event::bind_storage & ev) const noexcept {
    return ev.tensors != nullptr && ev.tensor_count > 0;
  }
};

struct invalid_bind {
  bool operator()(const event::bind_storage & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct valid_plan {
  bool operator()(const event::plan_load & ev) const noexcept {
    return ev.effect_capacity == 0 || ev.effects_out != nullptr;
  }
};

struct invalid_plan {
  bool operator()(const event::plan_load & ev) const noexcept {
    return !valid_plan{}(ev);
  }
};

struct valid_apply {
  bool operator()(const event::apply_effect_results & ev) const noexcept {
    return ev.result_count == 0 || ev.results != nullptr;
  }
};

struct invalid_apply {
  bool operator()(const event::apply_effect_results & ev) const noexcept {
    return !valid_apply{}(ev);
  }
};

}  // namespace emel::model::weight_loader::guard
