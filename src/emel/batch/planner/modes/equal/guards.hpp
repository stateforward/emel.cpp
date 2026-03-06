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
