#pragma once

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::equal::action {

using context = emel::batch::planner::action::context;
inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  detail::create_equal_plan(ev);
}

inline void create_plan_primary_fast_path_impl(const event::request_runtime & ev) noexcept {
  detail::create_equal_plan_primary_fast_path(ev);
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

inline constexpr auto prepare_steps = [](const event::request_runtime & ev, context &) noexcept {
  detail::prepare_plan(ev);
};

}  // namespace emel::batch::planner::modes::equal::action
