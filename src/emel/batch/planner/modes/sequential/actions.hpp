#pragma once

#include <array>

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::sequential::action {

using context = emel::batch::planner::action::context;
inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  while (used_count < ev.request.n_tokens) {
    int32_t cur_idx = 0;
    while (cur_idx < ev.request.n_tokens && used[static_cast<size_t>(cur_idx)] != 0) {
      ++cur_idx;
    }

    int32_t chunk = 0;
    detail::seq_mask_t cur_mask = detail::normalized_seq_mask(ev.request, cur_idx);
    (void)detail::begin_step(ev.ctx);

    bool continue_chunk = true;
    while (continue_chunk) {
      used[static_cast<size_t>(cur_idx)] = 1;
      used_count += 1;
      chunk += 1;
      (void)detail::append_token_index(ev.ctx, cur_idx);

      const bool reached_step_size = chunk >= ev.ctx.effective_step_size;
      int32_t next_idx = cur_idx + 1;
      while (!reached_step_size && next_idx < ev.request.n_tokens &&
             (used[static_cast<size_t>(next_idx)] != 0 ||
              !detail::mask_is_subset(cur_mask, detail::normalized_seq_mask(ev.request, next_idx)))) {
        ++next_idx;
      }

      const bool has_candidate = !reached_step_size && next_idx < ev.request.n_tokens;
      const detail::seq_mask_t next_mask = detail::normalized_seq_mask(
          ev.request,
          detail::select_i32(has_candidate, next_idx, 0));

      cur_idx = detail::select_i32(has_candidate, next_idx, cur_idx);
      detail::copy_mask_if(cur_mask, next_mask, has_candidate);
      continue_chunk = !reached_step_size && has_candidate;
    }

    (void)detail::push_step_size(ev.ctx, chunk);
  }

  detail::finalize_token_offsets(ev.ctx);
}

inline constexpr auto prepare_steps = [](const event::request_runtime & ev, context &) noexcept {
  detail::prepare_plan(ev);
};

inline constexpr auto mark_invalid_step_size = [](const event::request_runtime & ev,
                                                  context &) noexcept {
  detail::fail_plan(ev, error::invalid_step_size);
};

inline constexpr auto mark_output_steps_full = [](const event::request_runtime & ev,
                                                  context &) noexcept {
  detail::fail_plan(ev, error::output_steps_full);
};

inline constexpr auto mark_output_indices_full = [](const event::request_runtime & ev,
                                                    context &) noexcept {
  detail::fail_plan(ev, error::output_indices_full);
};

inline constexpr auto create_plan = [](const event::request_runtime & ev, context &) noexcept {
  create_plan_impl(ev);
};

}  // namespace emel::batch::planner::modes::sequential::action
