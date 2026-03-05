#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::equal::action {

using context = emel::batch::planner::action::context;

inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    detail::fail_plan(ev, error::invalid_step_size);
    return;
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  while (used_count < ev.request.n_tokens) {
    struct group_state {
      detail::seq_mask_t mask = {};
    };
    std::array<group_state, emel::batch::planner::action::MAX_PLAN_STEPS> groups = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
      if (used[static_cast<size_t>(i)] != 0) {
        continue;
      }

      const detail::seq_mask_t mask = detail::normalized_seq_mask(ev.request, i);
      bool overlap = false;
      for (int32_t g = 0; g < group_count; ++g) {
        if (detail::mask_overlaps(groups[static_cast<size_t>(g)].mask, mask)) {
          overlap = true;
          break;
        }
      }
      if (overlap) {
        continue;
      }

      if (ev.request.equal_sequential && ev.request.seq_primary_ids != nullptr) {
        const int32_t primary = ev.request.seq_primary_ids[i];
        if (group_count > 0 && primary != last_primary + 1) {
          continue;
        }
        last_primary = primary;
      }

      groups[static_cast<size_t>(group_count)] = group_state{.mask = mask};
      group_count += 1;
      if (group_count > ev.ctx.effective_step_size) {
        break;
      }
    }

    if (group_count == 0) {
      detail::fail_plan(ev, error::planning_progress_stalled);
      return;
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

    const int32_t max_rows = ev.ctx.effective_step_size / group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);
    if (n_seq_tokens <= 0) {
      detail::fail_plan(ev, error::planning_progress_stalled);
      return;
    }

    if (!detail::begin_step(ev.ctx)) {
      detail::fail_plan(ev, error::output_steps_full);
      return;
    }

    for (int32_t g = 0; g < group_count; ++g) {
      int32_t remaining = n_seq_tokens;
      for (int32_t i = 0; i < ev.request.n_tokens && remaining > 0; ++i) {
        if (used[static_cast<size_t>(i)] != 0) {
          continue;
        }
        if (!detail::mask_equal(detail::normalized_seq_mask(ev.request, i),
                                groups[static_cast<size_t>(g)].mask)) {
          continue;
        }
        used[static_cast<size_t>(i)] = 1;
        used_count += 1;
        if (!detail::append_token_index(ev.ctx, i)) {
          detail::fail_plan(ev, error::output_indices_full);
          return;
        }
        remaining -= 1;
      }
      if (remaining != 0) {
        detail::fail_plan(ev, error::algorithm_failed);
        return;
      }
    }

    if (!detail::push_step_size(ev.ctx, n_seq_tokens * group_count)) {
      detail::fail_plan(ev, error::output_steps_full);
      return;
    }
  }

  detail::finalize_token_offsets(ev.ctx);
}

inline void create_plan_primary_fast_path_impl(const event::request_runtime & ev) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    detail::fail_plan(ev, error::invalid_step_size);
    return;
  }
  if (ev.request.seq_primary_ids == nullptr) {
    detail::fail_plan(ev, error::invalid_sequence_id);
    return;
  }

  const int32_t max_seq = ev.request.seq_mask_words * 64;
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_counts = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ + 1> seq_offsets = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_used = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_cursor = {};
  std::array<int32_t, emel::batch::planner::action::MAX_PLAN_STEPS> seq_indices = {};

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    if (seq_id < 0 || seq_id >= max_seq) {
      detail::fail_plan(ev, error::invalid_sequence_id);
      return;
    }
    seq_counts[static_cast<size_t>(seq_id)] += 1;
  }

  for (int32_t s = 0; s < max_seq; ++s) {
    seq_offsets[static_cast<size_t>(s + 1)] =
        seq_offsets[static_cast<size_t>(s)] + seq_counts[static_cast<size_t>(s)];
    seq_cursor[static_cast<size_t>(s)] = seq_offsets[static_cast<size_t>(s)];
  }

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const size_t slot = static_cast<size_t>(seq_id);
    const int32_t pos = seq_cursor[slot];
    if (pos < 0 || pos >= ev.request.n_tokens) {
      detail::fail_plan(ev, error::algorithm_failed);
      return;
    }
    seq_indices[static_cast<size_t>(pos)] = i;
    seq_cursor[slot] = pos + 1;
  }

  int32_t remaining = ev.request.n_tokens;
  while (remaining > 0) {
    std::array<uint8_t, emel::batch::planner::action::MAX_SEQ> group_used = {};
    std::array<int32_t, emel::batch::planner::action::MAX_SEQ> group_ids = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
      const int32_t seq_id = ev.request.seq_primary_ids[i];
      const size_t slot = static_cast<size_t>(seq_id);
      if (seq_used[slot] >= seq_counts[slot]) {
        continue;
      }
      if (group_used[slot] != 0) {
        continue;
      }
      if (ev.request.equal_sequential && group_count > 0 && seq_id != last_primary + 1) {
        continue;
      }
      group_used[slot] = 1;
      group_ids[static_cast<size_t>(group_count)] = seq_id;
      group_count += 1;
      last_primary = seq_id;
      if (group_count > ev.ctx.effective_step_size) {
        break;
      }
    }

    if (group_count == 0) {
      detail::fail_plan(ev, error::planning_progress_stalled);
      return;
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(detail::select_i32(seq_id >= 0, seq_id, 0));
      const int32_t avail = seq_counts[slot] - seq_used[slot];
      min_avail = std::min(min_avail, avail);
    }

    const int32_t max_rows = ev.ctx.effective_step_size / group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);
    if (n_seq_tokens <= 0) {
      detail::fail_plan(ev, error::planning_progress_stalled);
      return;
    }

    if (!detail::begin_step(ev.ctx)) {
      detail::fail_plan(ev, error::output_steps_full);
      return;
    }

    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(detail::select_i32(seq_id >= 0, seq_id, 0));
      const int32_t base = seq_offsets[slot] + seq_used[slot];

      for (int32_t i = 0; i < n_seq_tokens; ++i) {
        const int32_t idx = seq_indices[static_cast<size_t>(base + i)];
        if (!detail::append_token_index(ev.ctx, idx)) {
          detail::fail_plan(ev, error::output_indices_full);
          return;
        }
      }

      seq_used[slot] += n_seq_tokens;
      remaining -= n_seq_tokens;
    }

    if (!detail::push_step_size(ev.ctx, n_seq_tokens * group_count)) {
      detail::fail_plan(ev, error::output_steps_full);
      return;
    }
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
