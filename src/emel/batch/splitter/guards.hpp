#pragma once

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"

namespace emel::batch::splitter::guard {

// validates callback contract on the triggering event.
inline constexpr auto callbacks_are_valid = [](const event::split & ev) noexcept {
  return static_cast<bool>(ev.on_done) && static_cast<bool>(ev.on_error);
};

inline constexpr auto callbacks_are_invalid =
    [](const event::split & ev) noexcept { return !callbacks_are_valid(ev); };

// validates request inputs copied into context (token pointer, counts, and mode).
inline constexpr auto inputs_are_valid = [](const action::context & ctx) noexcept {
  if (ctx.token_ids == nullptr || ctx.n_tokens <= 0) {
    return false;
  }
  if (ctx.n_tokens > action::MAX_UBATCHES) {
    return false;
  }
  if (ctx.seq_mask_words <= 0 || ctx.seq_mask_words > action::SEQ_WORDS) {
    return false;
  }
  if (ctx.output_mask != nullptr && ctx.output_mask_count < ctx.n_tokens) {
    return false;
  }
  if (ctx.seq_masks != nullptr && ctx.seq_masks_count < ctx.n_tokens) {
    return false;
  }
  if (ctx.seq_primary_ids != nullptr && ctx.seq_primary_ids_count < ctx.n_tokens) {
    return false;
  }

  if (ctx.seq_primary_ids != nullptr) {
    const int32_t max_seq = ctx.seq_mask_words * 64;
    for (int32_t i = 0; i < ctx.n_tokens; ++i) {
      const int32_t primary = ctx.seq_primary_ids[i];
      if (primary < 0 || primary >= max_seq) {
        return false;
      }
    }
  }

  if (ctx.seq_masks != nullptr) {
    for (int32_t i = 0; i < ctx.n_tokens; ++i) {
      const action::seq_mask_t mask = action::normalized_seq_mask(ctx, i);
      if (!action::mask_any_set(mask)) {
        return false;
      }
    }
  }

  if (ctx.mode == event::split_mode::equal && ctx.equal_sequential) {
    if (ctx.seq_masks != nullptr && ctx.seq_primary_ids == nullptr) {
      return false;
    }
    if (ctx.seq_primary_ids != nullptr) {
      for (int32_t i = 0; i < ctx.n_tokens; ++i) {
        const action::seq_mask_t mask = action::normalized_seq_mask(ctx, i);
        if (action::mask_has_multiple_bits(mask)) {
          return false;
        }
      }
    }
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

inline constexpr auto mode_is_equal_primary_fast = [](const action::context & ctx) noexcept {
  return ctx.mode == event::split_mode::equal && ctx.seq_masks == nullptr &&
         ctx.seq_primary_ids != nullptr;
};

inline constexpr auto mode_is_seq = [](const action::context & ctx) noexcept {
  return ctx.mode == event::split_mode::seq;
};

// reports whether split computation produced usable output sizes.
inline constexpr auto split_succeeded = [](const action::context & ctx) noexcept {
  if (ctx.ubatch_count <= 0 || ctx.total_outputs < 0) {
    return false;
  }
  if (ctx.token_indices_count != ctx.n_tokens) {
    return false;
  }
  if (ctx.ubatch_count <= action::MAX_UBATCHES) {
    return ctx.ubatch_token_offsets[ctx.ubatch_count] == ctx.token_indices_count;
  }
  return false;
};

inline constexpr auto split_failed =
    [](const action::context & ctx) noexcept { return !split_succeeded(ctx); };

}  // namespace emel::batch::splitter::guard
