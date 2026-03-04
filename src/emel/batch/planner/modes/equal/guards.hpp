#pragma once

#include <algorithm>
#include <array>

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::equal::guard {

inline constexpr auto mode_is_primary_fast_path =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.request.seq_masks == nullptr && ev.request.seq_primary_ids != nullptr;
    };

inline constexpr auto mode_is_general_path =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !mode_is_primary_fast_path(ev, ctx);
    };

inline constexpr auto has_valid_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.ctx.effective_step_size > 0;
    };

inline constexpr auto has_invalid_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !has_valid_step_size(ev, ctx);
    };

inline constexpr auto fast_path_has_primary_ids =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.request.seq_primary_ids != nullptr;
    };

inline constexpr auto fast_path_missing_primary_ids =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !fast_path_has_primary_ids(ev, ctx);
    };

inline int32_t available_step_slots(const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (ev.ctx.step_count < 0 || ev.ctx.step_count > emel::batch::planner::action::MAX_PLAN_STEPS) {
    return 0;
  }
  return emel::batch::planner::action::MAX_PLAN_STEPS - ev.ctx.step_count;
}

inline int32_t available_index_slots(const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (ev.ctx.token_indices_count < 0 ||
      ev.ctx.token_indices_count > emel::batch::planner::action::MAX_PLAN_STEPS) {
    return 0;
  }
  return emel::batch::planner::action::MAX_PLAN_STEPS - ev.ctx.token_indices_count;
}

inline constexpr auto has_step_capacity =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return available_step_slots(ev) > 0;
    };

inline constexpr auto lacks_step_capacity =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !has_step_capacity(ev, ctx);
    };

inline constexpr auto has_index_capacity =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.request.n_tokens <= available_index_slots(ev);
    };

inline constexpr auto lacks_index_capacity =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !has_index_capacity(ev, ctx);
    };

inline bool fast_path_primary_ids_valid_impl(
    const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (ev.request.seq_primary_ids == nullptr) {
    return false;
  }

  const int32_t max_seq = ev.request.seq_mask_words * 64;
  if (max_seq <= 0 || max_seq > emel::batch::planner::action::MAX_SEQ) {
    return false;
  }

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    if (seq_id < 0 || seq_id >= max_seq) {
      return false;
    }
  }

  return true;
}

inline constexpr auto fast_path_primary_ids_valid =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return fast_path_primary_ids_valid_impl(ev);
    };

inline constexpr auto fast_path_primary_ids_invalid =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !fast_path_primary_ids_valid(ev, ctx);
    };

inline bool general_first_group_scan_exceeds_step_size_impl(
    const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    return false;
  }
  if (ev.request.n_tokens <= 0) {
    return false;
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};

  struct group_state {
    emel::batch::planner::modes::detail::seq_mask_t mask = {};
  };
  std::array<group_state, emel::batch::planner::action::MAX_PLAN_STEPS> groups = {};
  int32_t group_count = 0;
  int32_t last_primary = -1;

  const int32_t primary_sink = 0;
  const std::array<const int32_t *, 2> primary_ptrs = {&primary_sink, ev.request.seq_primary_ids};

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const bool is_unused = used[static_cast<size_t>(i)] == 0;
    const auto mask = emel::batch::planner::modes::detail::normalized_seq_mask(ev.request, i);

    bool overlap = false;
    for (int32_t g = 0; g < group_count; ++g) {
      overlap = overlap ||
          emel::batch::planner::modes::detail::mask_overlaps(
              groups[static_cast<size_t>(g)].mask, mask);
    }

    const bool requires_sequential_primary =
        ev.request.equal_sequential && ev.request.seq_primary_ids != nullptr;
    const int32_t primary_read = primary_ptrs[static_cast<size_t>(requires_sequential_primary)]
        [static_cast<size_t>(emel::batch::planner::modes::detail::select_i32(
            requires_sequential_primary,
            i,
            0))];
    const int32_t primary = emel::batch::planner::modes::detail::select_i32(
        requires_sequential_primary,
        primary_read,
        last_primary);

    const bool out_of_order =
        requires_sequential_primary && group_count > 0 && primary != last_primary + 1;
    const bool can_add_group = is_unused && !overlap && !out_of_order;
    if (can_add_group) {
      if (group_count >= emel::batch::planner::action::MAX_PLAN_STEPS) {
        return true;
      }
      groups[static_cast<size_t>(group_count)].mask = mask;
      group_count += 1;
      if (requires_sequential_primary) {
        last_primary = primary;
      }
    }
  }

  return group_count > ev.ctx.effective_step_size;
}

inline bool fast_path_first_group_scan_exceeds_step_size_impl(
    const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (!fast_path_primary_ids_valid_impl(ev)) {
    return false;
  }
  if (ev.ctx.effective_step_size <= 0) {
    return false;
  }
  if (ev.request.n_tokens <= 0) {
    return false;
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_SEQ> group_used = {};
  int32_t group_count = 0;
  int32_t last_primary = -1;

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const size_t slot = static_cast<size_t>(seq_id);

    const bool already_grouped = group_used[slot] != 0;
    const bool out_of_order =
        ev.request.equal_sequential && group_count > 0 && seq_id != last_primary + 1;
    const bool use_slot = !already_grouped && !out_of_order;
    if (use_slot) {
      group_used[slot] = 1U;
      group_count += 1;
      last_primary = seq_id;
    }
  }

  return group_count > ev.ctx.effective_step_size;
}

inline constexpr auto general_first_group_scan_exceeds_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return general_first_group_scan_exceeds_step_size_impl(ev);
    };

inline constexpr auto general_first_group_scan_within_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !general_first_group_scan_exceeds_step_size(ev, ctx);
    };

inline constexpr auto fast_path_first_group_scan_exceeds_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return fast_path_first_group_scan_exceeds_step_size_impl(ev);
    };

inline constexpr auto fast_path_first_group_scan_within_step_size =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !fast_path_first_group_scan_exceeds_step_size(ev, ctx);
    };

inline bool general_progress_modelable_impl(
    const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (ev.ctx.effective_step_size <= 0) {
    return false;
  }
  if (ev.request.n_tokens <= 0) {
    return false;
  }
  if (available_step_slots(ev) <= 0) {
    return false;
  }
  if (ev.request.n_tokens > available_index_slots(ev)) {
    return false;
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;
  int32_t used_steps = 0;

  const int32_t primary_sink = 0;
  const std::array<const int32_t *, 2> primary_ptrs = {&primary_sink, ev.request.seq_primary_ids};

  while (used_count < ev.request.n_tokens) {
    struct group_state {
      emel::batch::planner::modes::detail::seq_mask_t mask = {};
    };

    std::array<group_state, emel::batch::planner::action::MAX_PLAN_STEPS> groups = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
      const bool is_unused = used[static_cast<size_t>(i)] == 0;
      const auto mask = emel::batch::planner::modes::detail::normalized_seq_mask(ev.request, i);

      bool overlap = false;
      for (int32_t g = 0; g < group_count; ++g) {
        overlap = overlap ||
            emel::batch::planner::modes::detail::mask_overlaps(
                groups[static_cast<size_t>(g)].mask, mask);
      }

      const bool requires_sequential_primary =
          ev.request.equal_sequential && ev.request.seq_primary_ids != nullptr;
      const int32_t primary_read = primary_ptrs[static_cast<size_t>(requires_sequential_primary)]
          [static_cast<size_t>(emel::batch::planner::modes::detail::select_i32(
              requires_sequential_primary,
              i,
              0))];
      const int32_t primary = emel::batch::planner::modes::detail::select_i32(
          requires_sequential_primary,
          primary_read,
          last_primary);

      const bool out_of_order =
          requires_sequential_primary && group_count > 0 && primary != last_primary + 1;
      const bool can_add_group = is_unused && !overlap && !out_of_order;
      if (can_add_group) {
        groups[static_cast<size_t>(group_count)].mask = mask;
        group_count += 1;
        if (requires_sequential_primary) {
          last_primary = primary;
        }
      }
    }

    if (group_count == 0) {
      return false;
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      int32_t avail = 0;
      for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
        const bool available =
            used[static_cast<size_t>(i)] == 0 &&
            emel::batch::planner::modes::detail::mask_equal(
                emel::batch::planner::modes::detail::normalized_seq_mask(ev.request, i),
                groups[static_cast<size_t>(g)].mask);
        avail += static_cast<int32_t>(available);
      }
      min_avail = std::min(min_avail, avail);
    }

    const int32_t max_rows = ev.ctx.effective_step_size / group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);
    if (n_seq_tokens <= 0) {
      return false;
    }

    used_steps += 1;
    if (used_steps > available_step_slots(ev)) {
      return false;
    }

    for (int32_t g = 0; g < group_count; ++g) {
      int32_t remaining = n_seq_tokens;
      for (int32_t i = 0; i < ev.request.n_tokens && remaining > 0; ++i) {
        const bool match =
            used[static_cast<size_t>(i)] == 0 &&
            emel::batch::planner::modes::detail::mask_equal(
                emel::batch::planner::modes::detail::normalized_seq_mask(ev.request, i),
                groups[static_cast<size_t>(g)].mask);
        if (match) {
          used[static_cast<size_t>(i)] = 1;
          used_count += 1;
          remaining -= 1;
        }
      }

      if (remaining != 0) {
        return false;
      }
    }
  }

  return true;
}

inline bool fast_path_progress_modelable_impl(
    const emel::batch::planner::event::request_runtime & ev) noexcept {
  if (!fast_path_primary_ids_valid_impl(ev)) {
    return false;
  }
  if (ev.ctx.effective_step_size <= 0) {
    return false;
  }
  if (ev.request.n_tokens <= 0) {
    return false;
  }
  if (available_step_slots(ev) <= 0) {
    return false;
  }
  if (ev.request.n_tokens > available_index_slots(ev)) {
    return false;
  }

  const int32_t max_seq = ev.request.seq_mask_words * 64;
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_counts = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ + 1> seq_offsets = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_used = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_cursor = {};
  std::array<int32_t, emel::batch::planner::action::MAX_PLAN_STEPS> seq_indices = {};

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
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
      return false;
    }
    seq_indices[static_cast<size_t>(pos)] = i;
    seq_cursor[slot] = pos + 1;
  }

  int32_t remaining = ev.request.n_tokens;
  int32_t used_steps = 0;

  while (remaining > 0) {
    std::array<uint8_t, emel::batch::planner::action::MAX_SEQ> group_used = {};
    std::array<int32_t, emel::batch::planner::action::MAX_SEQ> group_ids = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
      const int32_t seq_id = ev.request.seq_primary_ids[i];
      const size_t slot = static_cast<size_t>(seq_id);

      const bool slot_exhausted = seq_used[slot] >= seq_counts[slot];
      const bool already_grouped = group_used[slot] != 0;
      const bool out_of_order =
          ev.request.equal_sequential && group_count > 0 && seq_id != last_primary + 1;
      const bool use_slot = !slot_exhausted && !already_grouped && !out_of_order;
      if (use_slot) {
        group_used[slot] = 1U;
        group_ids[static_cast<size_t>(group_count)] = seq_id;
        group_count += 1;
        last_primary = seq_id;
      }
    }

    if (group_count == 0) {
      return false;
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(seq_id);
      const int32_t avail = seq_counts[slot] - seq_used[slot];
      min_avail = std::min(min_avail, avail);
    }

    const int32_t max_rows = ev.ctx.effective_step_size / group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);
    if (n_seq_tokens <= 0) {
      return false;
    }

    used_steps += 1;
    if (used_steps > available_step_slots(ev)) {
      return false;
    }

    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(seq_id);
      const int32_t base = seq_offsets[slot] + seq_used[slot];

      for (int32_t i = 0; i < n_seq_tokens; ++i) {
        const int32_t read_pos = base + i;
        if (read_pos < 0 || read_pos >= ev.request.n_tokens) {
          return false;
        }
        (void)seq_indices[static_cast<size_t>(read_pos)];
      }

      seq_used[slot] += n_seq_tokens;
      remaining -= n_seq_tokens;
    }
  }

  return true;
}

inline constexpr auto general_progress_modelable =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return general_progress_modelable_impl(ev);
    };

inline constexpr auto general_progress_not_modelable =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !general_progress_modelable(ev, ctx);
    };

inline constexpr auto fast_path_progress_modelable =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return fast_path_progress_modelable_impl(ev);
    };

inline constexpr auto fast_path_progress_not_modelable =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return !fast_path_progress_modelable(ev, ctx);
    };

inline constexpr auto fast_path_input_valid =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return has_valid_step_size(ev, ctx) &&
             fast_path_has_primary_ids(ev, ctx) &&
             fast_path_primary_ids_valid(ev, ctx);
    };

inline constexpr auto general_input_valid =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return has_valid_step_size(ev, ctx);
    };

inline constexpr auto storage_capacity_valid =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context & ctx) noexcept {
      return has_step_capacity(ev, ctx) && has_index_capacity(ev, ctx);
    };

inline constexpr auto planning_succeeded = [](const emel::batch::planner::event::request_runtime & ev,
                                              const emel::batch::planner::action::context &) noexcept {
  return emel::batch::planner::guard::planning_succeeded_impl(ev);
};

inline constexpr auto planning_failed = [](const emel::batch::planner::event::request_runtime & ev,
                                           const emel::batch::planner::action::context &) noexcept {
  return !emel::batch::planner::guard::planning_succeeded_impl(ev);
};

}  // namespace emel::batch::planner::modes::equal::guard
