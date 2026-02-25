#pragma once

#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/guards.hpp"

namespace emel::batch::planner::modes::equal::guard {

inline constexpr auto mode_is_primary_fast_path =
    [](const emel::batch::planner::event::request & ev,
       const emel::batch::planner::action::context &) noexcept {
      return ev.seq_masks == nullptr && ev.seq_primary_ids != nullptr;
    };

inline constexpr auto planning_succeeded = [](const emel::batch::planner::event::request & ev,
                                              const emel::batch::planner::action::context & ctx) noexcept {
  return emel::batch::planner::guard::planning_succeeded(ev, ctx);
};

inline constexpr auto planning_failed = [](const emel::batch::planner::event::request & ev,
                                           const emel::batch::planner::action::context & ctx) noexcept {
  return emel::batch::planner::guard::planning_failed(ev, ctx);
};

}  // namespace emel::batch::planner::modes::equal::guard
