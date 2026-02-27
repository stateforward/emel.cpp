#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::sequential::action {

using context = emel::batch::planner::action::context;
using seq_mask_t = detail::seq_mask_t;

inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    detail::fail_plan(ev, emel::batch::planner::error::invalid_step_size);
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
    seq_mask_t cur_mask = detail::normalized_seq_mask(ev.request, cur_idx);
    if (!detail::begin_step(ev.ctx)) {
      detail::fail_plan(ev, emel::batch::planner::error::output_steps_full);
      return;
    }
    while (true) {
      used[static_cast<size_t>(cur_idx)] = 1;
      used_count += 1;
      chunk += 1;
      if (!detail::append_token_index(ev.ctx, cur_idx)) {
        detail::fail_plan(ev, emel::batch::planner::error::output_indices_full);
        return;
      }

      if (chunk >= ev.ctx.effective_step_size) {
        break;
      }

      int32_t next_idx = cur_idx + 1;
      while (next_idx < ev.request.n_tokens) {
        if (used[static_cast<size_t>(next_idx)] == 0) {
          const seq_mask_t next_mask = detail::normalized_seq_mask(ev.request, next_idx);
          if (detail::mask_is_subset(cur_mask, next_mask)) {
            break;
          }
        }
        ++next_idx;
      }
      if (next_idx >= ev.request.n_tokens) {
        break;
      }

      cur_idx = next_idx;
      cur_mask = detail::normalized_seq_mask(ev.request, cur_idx);
    }

    if (!detail::push_step_size(ev.ctx, chunk)) {
      detail::fail_plan(ev, emel::batch::planner::error::output_steps_full);
      return;
    }
  }
  detail::finalize_token_offsets(ev.ctx);
}

inline constexpr auto prepare_steps = [](const event::request_runtime & ev, context &) noexcept {
  detail::prepare_plan(ev);
};

inline constexpr auto create_plan = [](const event::request_runtime & ev, context &) noexcept {
  create_plan_impl(ev);
};

}  // namespace emel::batch::planner::modes::sequential::action
