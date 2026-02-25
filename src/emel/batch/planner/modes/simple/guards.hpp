#pragma once

#include "emel/batch/planner/context.hpp"

namespace emel::batch::planner::modes::simple::guard {

inline constexpr auto planning_succeeded = [](const emel::batch::planner::event::request & ev,
                                              const emel::batch::planner::action::context & ctx) noexcept {
  if (ctx.step_count <= 0 || ctx.total_outputs < 0) {
    return false;
  }
  if (ctx.token_indices_count != ev.n_tokens) {
    return false;
  }
  if (ctx.step_count <= emel::batch::planner::action::MAX_PLAN_STEPS) {
    return ctx.step_token_offsets[ctx.step_count] == ctx.token_indices_count;
  }
  return false;
};

inline constexpr auto planning_failed =
    [](const emel::batch::planner::event::request & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !planning_succeeded(ev, ctx);
    };

}  // namespace emel::batch::planner::modes::simple::guard
