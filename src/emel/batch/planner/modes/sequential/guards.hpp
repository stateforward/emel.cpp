#pragma once

#include "emel/batch/planner/guards.hpp"
#include "emel/batch/planner/modes/sequential/context.hpp"
#include "emel/batch/planner/modes/sequential/events.hpp"

namespace emel::batch::planner::modes::sequential::guard {

inline int32_t compute_minimum_step_count(const event::plan_runtime & ev) noexcept {
  const int32_t step_size = ev.ctx.effective_step_size;
  if (step_size <= 0) {
    return 0;
  }
  const int32_t full_chunks = ev.request.n_tokens / step_size;
  const int32_t has_remainder = static_cast<int32_t>((ev.request.n_tokens % step_size) != 0);
  return full_chunks + has_remainder;
}

inline constexpr auto guard_has_valid_step_size =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return ev.ctx.effective_step_size > 0;
    };

inline constexpr auto guard_has_invalid_step_size =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_has_valid_step_size(ev, ctx);
    };

inline constexpr auto guard_exceeds_step_capacity =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return compute_minimum_step_count(ev) > emel::batch::planner::action::MAX_PLAN_STEPS;
    };

inline constexpr auto guard_exceeds_index_capacity =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return ev.request.n_tokens > emel::batch::planner::action::MAX_PLAN_STEPS;
    };

inline constexpr auto guard_sequential_plan_capacity_ok =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return guard_has_valid_step_size(ev, ctx) &&
             !guard_exceeds_step_capacity(ev, ctx) &&
             !guard_exceeds_index_capacity(ev, ctx);
    };

inline constexpr auto guard_planning_succeeded =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return emel::batch::planner::guard::guard_has_complete_plan(ev);
    };

inline constexpr auto guard_planning_failed = [](const event::plan_runtime & ev,
                                           const context &) noexcept {
  return !emel::batch::planner::guard::guard_has_complete_plan(ev);
};

}  // namespace emel::batch::planner::modes::sequential::guard
