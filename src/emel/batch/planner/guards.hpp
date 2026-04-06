#pragma once

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/detail.hpp"

namespace emel::batch::planner::guard {

inline bool guard_has_complete_plan(const event::plan_runtime & ev) noexcept {
  if (ev.ctx.err != emel::error::cast(error::none)) {
    return false;
  }
  if (ev.ctx.step_count <= 0 || ev.ctx.total_outputs < 0) {
    return false;
  }
  if (ev.ctx.token_indices_count != ev.request.n_tokens) {
    return false;
  }
  if (ev.ctx.step_count > action::MAX_PLAN_STEPS) {
    return false;
  }
  return ev.ctx.step_token_offsets[ev.ctx.step_count] == ev.ctx.token_indices_count;
}

inline constexpr auto guard_inputs_valid = [](const event::plan_runtime & ev,
                                              const action::context &) noexcept {
  return detail::has_input_errors(ev.request) == false;
};

inline constexpr auto guard_inputs_invalid = [](const event::plan_runtime & ev,
                                                const action::context & ctx) noexcept {
  return !guard_inputs_valid(ev, ctx);
};

inline constexpr auto guard_mode_is_simple = [](const event::plan_runtime & ev,
                                                const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::simple;
};

inline constexpr auto guard_mode_is_equal = [](const event::plan_runtime & ev,
                                               const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::equal;
};

inline constexpr auto guard_mode_is_sequential = [](const event::plan_runtime & ev,
                                                    const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::seq ||
         ev.request.mode == event::plan_mode::sequential;
};

inline constexpr auto guard_mode_is_invalid = [](const event::plan_runtime & ev,
                                                 const action::context & ctx) noexcept {
  return !guard_mode_is_simple(ev, ctx) && !guard_mode_is_equal(ev, ctx) &&
         !guard_mode_is_sequential(ev, ctx);
};

inline constexpr auto guard_planning_succeeded = [](const event::plan_runtime & ev,
                                                    const action::context &) noexcept {
  return guard_has_complete_plan(ev);
};

inline constexpr auto guard_planning_failed = [](const event::plan_runtime & ev,
                                                 const action::context &) noexcept {
  return !guard_has_complete_plan(ev);
};

inline constexpr auto guard_plan_error_present = [](const event::plan_runtime & ev,
                                                    const action::context &) noexcept {
  return ev.ctx.err != emel::error::cast(error::none);
};

inline constexpr auto guard_plan_error_absent = [](const event::plan_runtime & ev,
                                                   const action::context &) noexcept {
  return ev.ctx.err == emel::error::cast(error::none);
};

inline constexpr auto guard_planning_failed_with_error =
    [](const event::plan_runtime & ev, const action::context & ctx) noexcept {
  return guard_planning_failed(ev, ctx) && guard_plan_error_present(ev, ctx);
};

inline constexpr auto guard_planning_failed_without_error =
    [](const event::plan_runtime & ev, const action::context & ctx) noexcept {
  return guard_planning_failed(ev, ctx) && guard_plan_error_absent(ev, ctx);
};

}  // namespace emel::batch::planner::guard
