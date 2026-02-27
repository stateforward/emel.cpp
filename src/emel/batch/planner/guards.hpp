#pragma once

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::guard {

inline bool planning_succeeded_impl(const event::request_runtime & ev) noexcept {
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

inline constexpr auto inputs_are_valid = [](const event::request_runtime & ev,
                                            const action::context &) noexcept {
  return modes::detail::has_input_errors(ev.request) == false;
};

inline constexpr auto inputs_are_invalid = [](const event::request_runtime & ev,
                                              const action::context & ctx) noexcept {
  return !inputs_are_valid(ev, ctx);
};

inline constexpr auto mode_is_simple = [](const event::request_runtime & ev,
                                          const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::simple;
};

inline constexpr auto mode_is_equal = [](const event::request_runtime & ev,
                                         const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::equal;
};

inline constexpr auto mode_is_equal_primary_fast = [](const event::request_runtime & ev,
                                                      const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::equal && ev.request.seq_masks == nullptr &&
         ev.request.seq_primary_ids != nullptr;
};

inline constexpr auto mode_is_seq = [](const event::request_runtime & ev,
                                       const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::seq ||
         ev.request.mode == event::plan_mode::sequential;
};

inline constexpr auto mode_is_invalid = [](const event::request_runtime & ev,
                                           const action::context & ctx) noexcept {
  return !mode_is_simple(ev, ctx) && !mode_is_equal(ev, ctx) && !mode_is_seq(ev, ctx);
};

inline constexpr auto mode_is_sequential = [](const event::request_runtime & ev,
                                              const action::context &) noexcept {
  return ev.request.mode == event::plan_mode::sequential ||
         ev.request.mode == event::plan_mode::seq;
};

inline constexpr auto planning_succeeded = [](const event::request_runtime & ev,
                                              const action::context &) noexcept {
  return planning_succeeded_impl(ev);
};

inline constexpr auto planning_failed = [](const event::request_runtime & ev,
                                           const action::context &) noexcept {
  return !planning_succeeded_impl(ev);
};

inline constexpr auto plan_error_present = [](const event::request_runtime & ev,
                                              const action::context &) noexcept {
  return ev.ctx.err != emel::error::cast(error::none);
};

inline constexpr auto plan_error_absent = [](const event::request_runtime & ev,
                                             const action::context &) noexcept {
  return ev.ctx.err == emel::error::cast(error::none);
};

}  // namespace emel::batch::planner::guard
