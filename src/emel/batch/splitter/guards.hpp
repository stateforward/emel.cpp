#pragma once

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"

namespace emel::batch::splitter::guard {

// Validates callback contract on the triggering event.
inline constexpr auto callbacks_are_valid = [](const event::split & ev) noexcept {
  return static_cast<bool>(ev.on_done) && static_cast<bool>(ev.on_error);
};

inline constexpr auto callbacks_are_invalid =
    [](const event::split & ev) noexcept { return !callbacks_are_valid(ev); };

// Validates request inputs copied into context (token pointer, counts, and mode).
inline constexpr auto inputs_are_valid = [](const action::context & ctx) noexcept {
  if (ctx.token_ids == nullptr || ctx.n_tokens <= 0) {
    return false;
  }
  if (ctx.n_tokens > action::MAX_UBATCHES) {
    return false;
  }

  switch (ctx.mode) {
    case event::split_mode::simple:
    case event::split_mode::equal:
    case event::split_mode::seq:
      return true;
  }
  return false;
};

inline constexpr auto inputs_are_invalid =
    [](const action::context & ctx) noexcept { return !inputs_are_valid(ctx); };

inline constexpr auto mode_is_simple = [](const action::context & ctx) noexcept {
  return ctx.mode == event::split_mode::simple;
};

inline constexpr auto mode_is_equal = [](const action::context & ctx) noexcept {
  return ctx.mode == event::split_mode::equal;
};

inline constexpr auto mode_is_seq = [](const action::context & ctx) noexcept {
  return ctx.mode == event::split_mode::seq;
};

// Reports whether split computation produced usable output sizes.
inline constexpr auto split_succeeded = [](const action::context & ctx) noexcept {
  return ctx.ubatch_count > 0 && ctx.total_outputs > 0;
};

inline constexpr auto split_failed =
    [](const action::context & ctx) noexcept { return !split_succeeded(ctx); };

}  // namespace emel::batch::splitter::guard
