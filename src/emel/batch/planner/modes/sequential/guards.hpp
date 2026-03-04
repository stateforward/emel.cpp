#pragma once

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/guards.hpp"

namespace emel::batch::planner::modes::sequential::guard {

inline int32_t minimum_step_count(const emel::batch::planner::event::request_runtime & ev) noexcept {
  const int32_t step_size = ev.ctx.effective_step_size;
  if (step_size <= 0) {
    return 0;
  }
  const int32_t full_chunks = ev.request.n_tokens / step_size;
  const int32_t has_remainder = static_cast<int32_t>((ev.request.n_tokens % step_size) != 0);
  return full_chunks + has_remainder;
}

inline constexpr auto has_valid_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.ctx.effective_step_size > 0;
    };

inline constexpr auto has_invalid_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !has_valid_step_size(ev, ctx);
    };

inline constexpr auto exceeds_step_capacity =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return minimum_step_count(ev) > emel::batch::planner::action::MAX_PLAN_STEPS;
    };

inline constexpr auto exceeds_index_capacity =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.request.n_tokens > emel::batch::planner::action::MAX_PLAN_STEPS;
    };

inline constexpr auto sequential_plan_capacity_ok =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return has_valid_step_size(ev, ctx) &&
             !exceeds_step_capacity(ev, ctx) &&
             !exceeds_index_capacity(ev, ctx);
    };

inline constexpr auto planning_succeeded =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return emel::batch::planner::guard::planning_succeeded_impl(ev);
    };

inline constexpr auto planning_failed = [](const emel::batch::planner::event::request_runtime & ev,
                                           const emel::batch::planner::action::context &) noexcept {
  return !emel::batch::planner::guard::planning_succeeded_impl(ev);
};

}  // namespace emel::batch::planner::modes::sequential::guard
