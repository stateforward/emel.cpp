#pragma once

#include <algorithm>
#include <array>

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::equal::action {

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

inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;
  int32_t loop_budget = emel::batch::planner::action::MAX_PLAN_STEPS;

  const int32_t primary_sink = 0;
  const std::array<const int32_t *, 2> primary_ptrs = {&primary_sink, ev.request.seq_primary_ids};

  while (used_count < ev.request.n_tokens && loop_budget > 0) {
    struct group_state {
      detail::seq_mask_t mask = {};
    };

    std::array<group_state, emel::batch::planner::action::MAX_PLAN_STEPS> groups = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
      const bool is_unused = used[static_cast<size_t>(i)] == 0;
      const detail::seq_mask_t mask = detail::normalized_seq_mask(ev.request, i);

      bool overlap = false;
      for (int32_t g = 0; g < group_count; ++g) {
        overlap = overlap || detail::mask_overlaps(groups[static_cast<size_t>(g)].mask, mask);
      }

      const bool requires_sequential_primary =
          ev.request.equal_sequential && ev.request.seq_primary_ids != nullptr;
      const int32_t primary_read = primary_ptrs[static_cast<size_t>(requires_sequential_primary)]
          [static_cast<size_t>(detail::select_i32(requires_sequential_primary, i, 0))];
      const int32_t primary = detail::select_i32(requires_sequential_primary, primary_read, last_primary);

      const bool out_of_order =
          requires_sequential_primary && group_count > 0 && primary != last_primary + 1;
      const bool can_add_group = is_unused && !overlap && !out_of_order;

      const int32_t group_index = detail::select_i32(can_add_group, group_count, 0);
      detail::copy_mask_if(groups[static_cast<size_t>(group_index)].mask, mask, can_add_group);

      group_count += static_cast<int32_t>(can_add_group);
      last_primary = detail::select_i32(can_add_group && requires_sequential_primary,
                                        primary,
                                        last_primary);
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      int32_t avail = 0;
      for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
        const bool available =
            used[static_cast<size_t>(i)] == 0 &&
            detail::mask_equal(detail::normalized_seq_mask(ev.request, i),
                               groups[static_cast<size_t>(g)].mask);
        avail += static_cast<int32_t>(available);
      }
      min_avail = std::min(min_avail, avail);
    }

    const int32_t safe_group_count = detail::select_i32(group_count > 0, group_count, 1);
    const int32_t max_rows = ev.ctx.effective_step_size / safe_group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);

    (void)detail::begin_step(ev.ctx);

    for (int32_t g = 0; g < group_count; ++g) {
      int32_t remaining = n_seq_tokens;
      for (int32_t i = 0; i < ev.request.n_tokens && remaining > 0; ++i) {
        const bool match =
            used[static_cast<size_t>(i)] == 0 &&
            detail::mask_equal(detail::normalized_seq_mask(ev.request, i),
                               groups[static_cast<size_t>(g)].mask);

        used[static_cast<size_t>(i)] =
            detail::select_u8(match, 1u, used[static_cast<size_t>(i)]);
        used_count += static_cast<int32_t>(match);
        append_token_index_if(ev.ctx, match, i);
        remaining -= static_cast<int32_t>(match);
      }
    }

    const int32_t added = n_seq_tokens * group_count;
    (void)detail::push_step_size(ev.ctx, added);
    loop_budget -= 1;
  }

  detail::finalize_token_offsets(ev.ctx);
}

inline void create_plan_primary_fast_path_impl(const event::request_runtime & ev) noexcept {
  const int32_t max_seq = ev.request.seq_mask_words * 64;
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_counts = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ + 1> seq_offsets = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_used = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_cursor = {};
  std::array<int32_t, emel::batch::planner::action::MAX_PLAN_STEPS> seq_indices = {};

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const bool valid_seq = seq_id >= 0 && seq_id < max_seq;
    const size_t slot = static_cast<size_t>(detail::select_i32(valid_seq, seq_id, 0));
    seq_counts[slot] += static_cast<int32_t>(valid_seq);
  }

  for (int32_t s = 0; s < max_seq; ++s) {
    seq_offsets[static_cast<size_t>(s + 1)] =
        seq_offsets[static_cast<size_t>(s)] + seq_counts[static_cast<size_t>(s)];
    seq_cursor[static_cast<size_t>(s)] = seq_offsets[static_cast<size_t>(s)];
  }

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const bool valid_seq = seq_id >= 0 && seq_id < max_seq;
    const size_t slot = static_cast<size_t>(detail::select_i32(valid_seq, seq_id, 0));

    const int32_t pos = seq_cursor[slot];
    const bool valid_pos = pos >= 0 && pos < ev.request.n_tokens;
    const size_t write_pos = static_cast<size_t>(detail::select_i32(valid_pos, pos, 0));
    seq_indices[write_pos] = detail::select_i32(valid_pos, i, seq_indices[write_pos]);
    seq_cursor[slot] = pos + static_cast<int32_t>(valid_pos);
  }

  int32_t remaining = ev.request.n_tokens;
  int32_t loop_budget = emel::batch::planner::action::MAX_PLAN_STEPS;
  while (remaining > 0 && loop_budget > 0) {
    std::array<uint8_t, emel::batch::planner::action::MAX_SEQ> group_used = {};
    std::array<int32_t, emel::batch::planner::action::MAX_SEQ> group_ids = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
      const int32_t seq_id = ev.request.seq_primary_ids[i];
      const bool valid_seq = seq_id >= 0 && seq_id < max_seq;
      const size_t slot = static_cast<size_t>(detail::select_i32(valid_seq, seq_id, 0));

      const bool slot_exhausted = !valid_seq || seq_used[slot] >= seq_counts[slot];
      const bool already_grouped = group_used[slot] != 0;
      const bool out_of_order =
          ev.request.equal_sequential && group_count > 0 && seq_id != last_primary + 1;
      const bool skip_slot = slot_exhausted || already_grouped || out_of_order;
      const bool use_slot = !skip_slot;

      group_used[slot] = detail::select_u8(use_slot, 1u, group_used[slot]);
      const int32_t group_index = detail::select_i32(use_slot, group_count, 0);
      group_ids[static_cast<size_t>(group_index)] =
          detail::select_i32(use_slot, seq_id, group_ids[static_cast<size_t>(group_index)]);
      group_count += static_cast<int32_t>(use_slot);
      last_primary = detail::select_i32(use_slot, seq_id, last_primary);
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(detail::select_i32(seq_id >= 0, seq_id, 0));
      const int32_t avail = seq_counts[slot] - seq_used[slot];
      min_avail = std::min(min_avail, avail);
    }

    const int32_t safe_group_count = detail::select_i32(group_count > 0, group_count, 1);
    const int32_t max_rows = ev.ctx.effective_step_size / safe_group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);

    (void)detail::begin_step(ev.ctx);

    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(detail::select_i32(seq_id >= 0, seq_id, 0));
      const int32_t base = seq_offsets[slot] + seq_used[slot];

      for (int32_t i = 0; i < n_seq_tokens; ++i) {
        const int32_t read_pos = base + i;
        const bool valid_read = read_pos >= 0 && read_pos < ev.request.n_tokens;
        const size_t read_index = static_cast<size_t>(detail::select_i32(valid_read, read_pos, 0));
        const int32_t idx = seq_indices[read_index];
        (void)detail::append_token_index(ev.ctx, idx);
      }

      seq_used[slot] += n_seq_tokens;
      remaining -= n_seq_tokens;
    }

    const int32_t added = n_seq_tokens * group_count;
    (void)detail::push_step_size(ev.ctx, added);
    loop_budget -= 1;
  }

  detail::finalize_token_offsets(ev.ctx);
}

inline constexpr auto create_plan = [](const event::request_runtime & ev, context &) noexcept {
  create_plan_impl(ev);
};

inline constexpr auto create_plan_primary_fast_path = [](const event::request_runtime & ev,
                                                         context &) noexcept {
  create_plan_primary_fast_path_impl(ev);
};

inline constexpr auto create_plan_general = [](const event::request_runtime & ev,
                                               context &) noexcept {
  create_plan_impl(ev);
};

inline constexpr auto mark_invalid_step_size = [](const event::request_runtime & ev,
                                                  context &) noexcept {
  detail::fail_plan(ev, error::invalid_step_size);
};

inline constexpr auto mark_invalid_sequence_id = [](const event::request_runtime & ev,
                                                    context &) noexcept {
  detail::fail_plan(ev, error::invalid_sequence_id);
};

inline constexpr auto mark_planning_progress_stalled = [](const event::request_runtime & ev,
                                                          context &) noexcept {
  detail::fail_plan(ev, error::planning_progress_stalled);
};

inline constexpr auto mark_output_steps_full = [](const event::request_runtime & ev,
                                                  context &) noexcept {
  detail::fail_plan(ev, error::output_steps_full);
};

inline constexpr auto mark_output_indices_full = [](const event::request_runtime & ev,
                                                    context &) noexcept {
  detail::fail_plan(ev, error::output_indices_full);
};

inline constexpr auto prepare_steps = [](const event::request_runtime & ev, context &) noexcept {
  detail::prepare_plan(ev);
};

}  // namespace emel::batch::planner::modes::equal::action
