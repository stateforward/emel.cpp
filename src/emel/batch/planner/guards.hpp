#pragma once

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/modes/detail.hpp"
#include "emel/batch/planner/events.hpp"

namespace emel::batch::planner::guard {

// validates request input payload (token pointer, counts, and metadata).
inline constexpr auto inputs_are_valid =
    [](const event::request & ev, const action::context &) noexcept {
      return modes::detail::has_input_errors(ev) == false;
    };

inline constexpr auto inputs_are_invalid =
    [](const event::request & ev, const action::context & ctx) noexcept {
      return !inputs_are_valid(ev, ctx);
    };

inline constexpr auto mode_is_simple =
    [](const event::request & ev, const action::context &) noexcept {
      return ev.mode == event::plan_mode::simple;
    };

inline constexpr auto mode_is_equal =
    [](const event::request & ev, const action::context &) noexcept {
      return ev.mode == event::plan_mode::equal;
    };

inline constexpr auto mode_is_equal_primary_fast =
    [](const event::request & ev, const action::context &) noexcept {
      return ev.mode == event::plan_mode::equal && ev.seq_masks == nullptr &&
             ev.seq_primary_ids != nullptr;
    };

inline constexpr auto mode_is_seq = [](const event::request & ev,
                                       const action::context &) noexcept {
  return ev.mode == event::plan_mode::seq ||
         ev.mode == event::plan_mode::sequential;
};

inline constexpr auto mode_is_invalid =
    [](const event::request & ev, const action::context & ctx) noexcept {
      return !mode_is_simple(ev, ctx) && !mode_is_equal(ev, ctx) && !mode_is_seq(ev, ctx);
    };

inline constexpr auto mode_is_sequential =
    [](const event::request & ev, const action::context &) noexcept {
      return ev.mode == event::plan_mode::sequential ||
             ev.mode == event::plan_mode::seq;
    };

// reports whether plan computation produced usable output sizes.
inline constexpr auto planning_succeeded =
    [](const event::request & ev, const action::context & ctx) noexcept {
      if (ctx.step_count <= 0 || ctx.total_outputs < 0) {
        return false;
      }
      if (ctx.token_indices_count != ev.n_tokens) {
        return false;
      }
      if (ctx.step_count <= action::MAX_PLAN_STEPS) {
        return ctx.step_token_offsets[ctx.step_count] == ctx.token_indices_count;
      }
      return false;
    };

inline constexpr auto planning_failed =
    [](const event::request & ev, const action::context & ctx) noexcept {
      return !planning_succeeded(ev, ctx);
    };

}  // namespace emel::batch::planner::guard
