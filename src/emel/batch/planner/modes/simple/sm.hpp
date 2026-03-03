#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/batch/planner/modes/simple/actions.hpp"
#include "emel/batch/planner/modes/simple/guards.hpp"

namespace emel::batch::planner::modes::simple {

struct preparing {};
struct planning {};
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
      , sml::state<planning_decision> <= sml::state<planning>
          + sml::completion<event::request_runtime> / action::create_plan
      , sml::state<planning_done> <= sml::state<planning_decision>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ]
      , sml::state<planning_failed> <= sml::state<planning_decision>
          + sml::completion<event::request_runtime> [ guard::planning_failed ]
      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<planning_done>
      , sml::X <= sml::state<planning_failed>
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_done>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_failed>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<preparing>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_decision>
          + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

struct sm : emel::sm<model> {
  using model_type = model;
};

}  // namespace emel::batch::planner::modes::simple
