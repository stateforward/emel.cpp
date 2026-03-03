#pragma once

#include "emel/batch/planner/modes/detail.hpp"

namespace emel::batch::planner::modes::sequential::action {

using context = emel::batch::planner::action::context;
inline void create_plan_impl(const event::request_runtime & ev) noexcept {
  detail::create_sequential_plan(ev);
}

inline constexpr auto prepare_steps = [](const event::request_runtime & ev, context &) noexcept {
  detail::prepare_plan(ev);
};

inline constexpr auto create_plan = [](const event::request_runtime & ev, context &) noexcept {
  create_plan_impl(ev);
};

}  // namespace emel::batch::planner::modes::sequential::action
