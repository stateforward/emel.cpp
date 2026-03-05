#pragma once

#include <array>

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::sequential::action {

using context = emel::batch::planner::action::context;

inline void append_token_index_if(event::request_ctx & ctx,
                                  const bool should_append,
                                  const int32_t idx) noexcept {
  using append_handler = void (*)(event::request_ctx &, int32_t) noexcept;
  static const std::array<append_handler, 2> handlers = {
      +[](event::request_ctx &, const int32_t) noexcept {},
      +[](event::request_ctx & inner_ctx, const int32_t token_idx) noexcept {
        (void)detail::append_token_index(inner_ctx, token_idx);
      },
  };
  handlers[static_cast<size_t>(should_append)](ctx, idx);
}

inline void begin_step_if(event::request_ctx & ctx, const bool should_begin) noexcept {
  using begin_handler = void (*)(event::request_ctx &) noexcept;
  static const std::array<begin_handler, 2> handlers = {
      +[](event::request_ctx &) noexcept {},
      +[](event::request_ctx & inner_ctx) noexcept {
        (void)detail::begin_step(inner_ctx);
      },
  };
  handlers[static_cast<size_t>(should_begin)](ctx);
}

inline void push_step_size_if(event::request_ctx & ctx,
                              const bool should_push,
                              const int32_t step_size) noexcept {
  using push_handler = void (*)(event::request_ctx &, int32_t) noexcept;
  static const std::array<push_handler, 2> handlers = {
      +[](event::request_ctx &, const int32_t) noexcept {},
      +[](event::request_ctx & inner_ctx, const int32_t inner_step_size) noexcept {
        (void)detail::push_step_size(inner_ctx, inner_step_size);
      },
  };
  handlers[static_cast<size_t>(should_push)](ctx, step_size);
}

inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  for (int32_t step_iteration = 0;
       step_iteration < emel::batch::planner::action::MAX_PLAN_STEPS;
       ++step_iteration) {
    const bool step_active = used_count < ev.request.n_tokens;

    int32_t cur_idx = 0;
    bool cur_idx_found = false;
    for (int32_t scan = 0; scan < ev.request.n_tokens; ++scan) {
      const bool can_pick =
          step_active && !cur_idx_found && used[static_cast<size_t>(scan)] == 0;
      cur_idx = detail::select_i32(can_pick, scan, cur_idx);
      cur_idx_found = cur_idx_found || can_pick;
    }

    int32_t chunk = 0;
    const bool chunk_active = step_active && cur_idx_found;
    detail::seq_mask_t cur_mask = detail::normalized_seq_mask(
        ev.request,
        detail::select_i32(chunk_active, cur_idx, 0));
    begin_step_if(ev.ctx, chunk_active);

    bool continue_chunk = chunk_active;
    for (int32_t row = 0; row < emel::batch::planner::action::MAX_PLAN_STEPS; ++row) {
      const bool append_current = continue_chunk;
      used[static_cast<size_t>(cur_idx)] =
          detail::select_u8(append_current, 1u, used[static_cast<size_t>(cur_idx)]);
      used_count += static_cast<int32_t>(append_current);
      chunk += static_cast<int32_t>(append_current);
      append_token_index_if(ev.ctx, append_current, cur_idx);

      const bool reached_step_size = chunk >= ev.ctx.effective_step_size;
      int32_t next_idx = cur_idx;
      bool has_candidate = false;
      for (int32_t scan = 0; scan < ev.request.n_tokens; ++scan) {
        const bool after_current = scan > cur_idx;
        const bool unused_scan = used[static_cast<size_t>(scan)] == 0;
        const bool subset_scan =
            detail::mask_is_subset(cur_mask, detail::normalized_seq_mask(ev.request, scan));
        const bool can_pick_next =
            append_current && !reached_step_size && !has_candidate &&
            after_current && unused_scan && subset_scan;
        next_idx = detail::select_i32(can_pick_next, scan, next_idx);
        has_candidate = has_candidate || can_pick_next;
      }

      const detail::seq_mask_t next_mask = detail::normalized_seq_mask(
          ev.request,
          detail::select_i32(has_candidate, next_idx, 0));

      cur_idx = detail::select_i32(has_candidate, next_idx, cur_idx);
      detail::copy_mask_if(cur_mask, next_mask, has_candidate);
      continue_chunk = append_current && !reached_step_size && has_candidate;
    }

    push_step_size_if(ev.ctx, chunk_active, chunk);
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

inline constexpr auto mark_planning_progress_stalled = [](const event::request_runtime & ev,
                                                          context &) noexcept {
  detail::fail_plan(ev, error::planning_progress_stalled);
};

inline constexpr auto create_plan = [](const event::request_runtime & ev, context &) noexcept {
  create_plan_impl(ev);
};

}  // namespace emel::batch::planner::modes::sequential::action
