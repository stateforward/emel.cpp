#pragma once

#include "emel/batch/planner/modes/sequential/context.hpp"
#include "emel/batch/planner/modes/sequential/detail.hpp"
#include "emel/batch/planner/modes/sequential/events.hpp"

namespace emel::batch::planner::modes::sequential::action {

using context = emel::batch::planner::action::context;

inline constexpr auto effect_plan_sequential_batches =
    [](const event::plan_runtime & ev, context &) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    detail::fail_plan(ev, error::invalid_step_size);
    return;
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  while (used_count < ev.request.n_tokens) {
    int32_t cur_idx = 0;
    while (cur_idx < ev.request.n_tokens && used[static_cast<size_t>(cur_idx)] != 0) {
      ++cur_idx;
    }
    if (cur_idx >= ev.request.n_tokens) {
      break;
    }

    int32_t chunk = 0;
    detail::seq_mask_t cur_mask = detail::normalized_seq_mask(ev.request, cur_idx);
    if (!detail::begin_step(ev.ctx)) {
      detail::fail_plan(ev, error::output_steps_full);
      return;
    }

    while (true) {
      used[static_cast<size_t>(cur_idx)] = 1;
      used_count += 1;
      chunk += 1;
      if (!detail::append_token_index(ev.ctx, cur_idx)) {
        detail::fail_plan(ev, error::output_indices_full);
        return;
      }

      if (chunk >= ev.ctx.effective_step_size) {
        break;
      }

      int32_t next_idx = cur_idx + 1;
      while (next_idx < ev.request.n_tokens) {
        if (used[static_cast<size_t>(next_idx)] == 0) {
          const detail::seq_mask_t next_mask = detail::normalized_seq_mask(ev.request, next_idx);
          if (detail::mask_is_subset(cur_mask, next_mask)) {
            cur_idx = next_idx;
            cur_mask = next_mask;
            break;
          }
        }
        ++next_idx;
      }
      if (next_idx >= ev.request.n_tokens) {
        break;
      }
    }

    if (!detail::push_step_size(ev.ctx, chunk)) {
      detail::fail_plan(ev, error::output_steps_full);
      return;
    }
  }

  detail::finalize_token_offsets(ev.ctx);
};

inline constexpr auto effect_begin_planning = [](const event::plan_runtime & ev,
                                                 context &) noexcept {
  detail::prepare_plan(ev);
};

inline constexpr auto effect_emit_plan_done = [](const event::plan_runtime & ev,
                                                 const context &) noexcept {
  detail::notify_mode_done<events::plan_done>(ev);
};

inline constexpr auto effect_reject_invalid_step_size =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::invalid_step_size);
  detail::notify_mode_error<events::plan_error>(ev, ev.ctx.err);
};

inline constexpr auto effect_reject_output_steps_full =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::output_steps_full);
  detail::notify_mode_error<events::plan_error>(ev, ev.ctx.err);
};

inline constexpr auto effect_reject_output_indices_full =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::output_indices_full);
  detail::notify_mode_error<events::plan_error>(ev, ev.ctx.err);
};

inline constexpr auto effect_reject_planning_progress_stalled =
    [](const event::plan_runtime & ev, context &) noexcept {
  detail::fail_plan(ev, error::planning_progress_stalled);
  detail::notify_mode_error<events::plan_error>(ev, ev.ctx.err);
};

inline constexpr auto effect_reject_unexpected_event = [](const auto & ev) noexcept {
  if constexpr (requires { ev.request; ev.ctx; ev.on_error; }) {
    ev.ctx.err = emel::error::set(ev.ctx.err, error::untracked);
    detail::notify_mode_error<events::plan_error>(ev, ev.ctx.err);
  }
};

}  // namespace emel::batch::planner::modes::sequential::action
