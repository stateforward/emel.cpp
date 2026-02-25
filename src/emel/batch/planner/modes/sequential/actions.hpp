#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/modes/detail.hpp"
#include "emel/batch/planner/context.hpp"

namespace emel::batch::planner::modes::sequential::action {

using context = emel::batch::planner::action::context;
using seq_mask_t = detail::seq_mask_t;

inline void create_plan_impl(const event::request & ev, context & ctx) noexcept {
  if (ctx.effective_step_size <= 0) {
    detail::fail_plan(ctx, emel::batch::planner::error::invalid_step_size);
    return;
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  while (used_count < ev.n_tokens) {
    int32_t cur_idx = 0;
    while (cur_idx < ev.n_tokens && used[static_cast<size_t>(cur_idx)] != 0) {
      ++cur_idx;
    }
    if (cur_idx >= ev.n_tokens) {
      break;
    }

    int32_t chunk = 0;
    seq_mask_t cur_mask = detail::normalized_seq_mask(ev, cur_idx);
    if (!detail::begin_step(ctx)) {
      detail::fail_plan(ctx,
                        emel::batch::planner::error::output_steps_full);
      return;
    }
    while (true) {
      used[static_cast<size_t>(cur_idx)] = 1;
      used_count += 1;
      chunk += 1;
      if (!detail::append_token_index(ctx, cur_idx)) {
        detail::fail_plan(
            ctx, emel::batch::planner::error::output_indices_full);
        return;
      }

      if (chunk >= ctx.effective_step_size) {
        break;
      }

      int32_t next_idx = cur_idx + 1;
      while (next_idx < ev.n_tokens) {
        if (used[static_cast<size_t>(next_idx)] == 0) {
          const seq_mask_t next_mask = detail::normalized_seq_mask(ev, next_idx);
          if (detail::mask_is_subset(cur_mask, next_mask)) {
            break;
          }
        }
        ++next_idx;
      }
      if (next_idx >= ev.n_tokens) {
        break;
      }

      cur_idx = next_idx;
      cur_mask = detail::normalized_seq_mask(ev, cur_idx);
    }

    if (!detail::push_step_size(ctx, chunk)) {
      detail::fail_plan(ctx, emel::batch::planner::error::output_steps_full);
      return;
    }
  }
  detail::finalize_token_offsets(ctx);
}

inline constexpr auto prepare_steps = [](const event::request & ev, context & ctx) noexcept {
  detail::prepare_plan(ev, ctx);
};

inline constexpr auto create_plan = [](const event::request & ev, context & ctx) noexcept {
  create_plan_impl(ev, ctx);
};

}  // namespace emel::batch::planner::modes::sequential::action
