#pragma once

#include "emel/batch/planner/guards.hpp"

namespace emel::batch::planner::modes::equal::guard {

inline constexpr auto mode_is_primary_fast_path =
    [](const emel::batch::planner::event::request_runtime & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.request.seq_masks == nullptr && ev.request.seq_primary_ids != nullptr;
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
