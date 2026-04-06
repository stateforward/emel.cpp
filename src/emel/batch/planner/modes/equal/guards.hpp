#pragma once

#include <algorithm>
#include <array>

#include "emel/batch/planner/guards.hpp"
#include "emel/batch/planner/modes/equal/context.hpp"
#include "emel/batch/planner/modes/equal/detail.hpp"
#include "emel/batch/planner/modes/equal/events.hpp"

namespace emel::batch::planner::modes::equal::guard {

inline constexpr auto guard_mode_is_primary_fast_path =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return ev.request.seq_masks == nullptr && ev.request.seq_primary_ids != nullptr;
    };

inline constexpr auto guard_mode_is_general_path =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_mode_is_primary_fast_path(ev, ctx);
    };

inline constexpr auto guard_has_valid_step_size =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return ev.ctx.effective_step_size > 0;
    };

inline constexpr auto guard_has_invalid_step_size =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_has_valid_step_size(ev, ctx);
    };

inline constexpr auto guard_fast_path_has_primary_ids =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return ev.request.seq_primary_ids != nullptr;
    };

inline constexpr auto guard_fast_path_missing_primary_ids =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_fast_path_has_primary_ids(ev, ctx);
    };

inline int32_t compute_available_step_slots(const event::plan_runtime & ev) noexcept {
  if (ev.ctx.step_count < 0 || ev.ctx.step_count > emel::batch::planner::action::MAX_PLAN_STEPS) {
    return 0;
  }
  return emel::batch::planner::action::MAX_PLAN_STEPS - ev.ctx.step_count;
}

inline int32_t compute_available_index_slots(const event::plan_runtime & ev) noexcept {
  if (ev.ctx.token_indices_count < 0 ||
      ev.ctx.token_indices_count > emel::batch::planner::action::MAX_PLAN_STEPS) {
    return 0;
  }
  return emel::batch::planner::action::MAX_PLAN_STEPS - ev.ctx.token_indices_count;
}

inline constexpr auto guard_has_step_capacity =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return compute_available_step_slots(ev) > 0;
    };

inline constexpr auto guard_lacks_step_capacity =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_has_step_capacity(ev, ctx);
    };

inline constexpr auto guard_has_index_capacity =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return ev.request.n_tokens <= compute_available_index_slots(ev);
    };

inline constexpr auto guard_lacks_index_capacity =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_has_index_capacity(ev, ctx);
    };

inline bool guard_fast_path_primary_ids_valid_impl(const event::plan_runtime & ev) noexcept {
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

inline constexpr auto guard_fast_path_primary_ids_valid =
    [](const event::plan_runtime & ev,
       const context &) noexcept {
      return guard_fast_path_primary_ids_valid_impl(ev);
    };

inline constexpr auto guard_fast_path_primary_ids_invalid =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return !guard_fast_path_primary_ids_valid(ev, ctx);
    };

inline constexpr auto guard_fast_path_input_valid =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return guard_has_valid_step_size(ev, ctx) &&
             guard_fast_path_has_primary_ids(ev, ctx) &&
             guard_fast_path_primary_ids_valid(ev, ctx);
    };

inline constexpr auto guard_general_input_valid =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return guard_has_valid_step_size(ev, ctx);
    };

inline constexpr auto guard_storage_capacity_valid =
    [](const event::plan_runtime & ev,
       const context & ctx) noexcept {
      return guard_has_step_capacity(ev, ctx) && guard_has_index_capacity(ev, ctx);
    };

inline constexpr auto guard_planning_succeeded = [](const event::plan_runtime & ev,
                                              const context &) noexcept {
  return emel::batch::planner::guard::guard_has_complete_plan(ev);
};

inline constexpr auto guard_planning_failed = [](const event::plan_runtime & ev,
                                           const context &) noexcept {
  return !emel::batch::planner::guard::guard_has_complete_plan(ev);
};

}  // namespace emel::batch::planner::modes::equal::guard
