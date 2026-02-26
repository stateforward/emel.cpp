#pragma once

#include "emel/model/weight_loader/context.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::action {

inline void set_error(context & ctx, int32_t err) noexcept {
  ctx.last_error = err;
}

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_INVALID_ARGUMENT); }
};

struct run_bind_storage {
  void operator()(const event::bind_storage & ev, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
    ctx.bound_ok = true;
    ctx.tensors = ev.tensors;
    ctx.tensor_count = ev.tensor_count;
  }
};

struct run_plan_load {
  void operator()(const event::plan_load & ev, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
    ctx.planned_effects = 0;
    if (ev.effect_count_out != nullptr) {
      *ev.effect_count_out = 0;
    }
  }
};

struct run_apply_effects {
  void operator()(const event::apply_effect_results &, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr run_bind_storage run_bind_storage{};
inline constexpr run_plan_load run_plan_load{};
inline constexpr run_apply_effects run_apply_effects{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::model::weight_loader::action
