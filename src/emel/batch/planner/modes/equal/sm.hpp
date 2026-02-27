#pragma once
// benchmark: scaffold

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/batch/planner/modes/equal/actions.hpp"
#include "emel/batch/planner/modes/equal/guards.hpp"

namespace emel::batch::planner::modes::equal {

struct preparing {};
struct planning {};
struct planning_mode_decision {};
struct planning_fast_path {};
struct planning_general {};
struct planning_decision {};
struct planning_done {};
struct planning_failed {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<planning> <= *sml::state<preparing>
          + sml::completion<event::request_runtime> / action::prepare_steps
      , sml::state<planning_mode_decision> <= sml::state<planning>
          + sml::completion<event::request_runtime>
      //------------------------------------------------------------------------------//
      , sml::state<planning_fast_path> <= sml::state<planning_mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_primary_fast_path ]
      , sml::state<planning_general> <= sml::state<planning_mode_decision>
          + sml::completion<event::request_runtime>
      //------------------------------------------------------------------------------//
      , sml::state<planning_decision> <= sml::state<planning_fast_path>
          + sml::completion<event::request_runtime> / action::create_plan_primary_fast_path
      , sml::state<planning_decision> <= sml::state<planning_general>
          + sml::completion<event::request_runtime> / action::create_plan_general
      //------------------------------------------------------------------------------//
      , sml::state<planning_done> <= sml::state<planning_decision>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ]
      , sml::state<planning_failed> <= sml::state<planning_decision>
          + sml::completion<event::request_runtime> [ guard::planning_failed ]
      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<planning_done>
      , sml::X <= sml::state<planning_failed>
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<preparing>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_mode_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_fast_path>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_done>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_failed>
          + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

struct sm : emel::sm<model> {
  using model_type = model;
};

}  // namespace emel::batch::planner::modes::equal
