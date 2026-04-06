#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/batch/planner/detail.hpp"
#include "emel/batch/planner/modes/simple/actions.hpp"
#include "emel/batch/planner/modes/simple/events.hpp"
#include "emel/batch/planner/modes/simple/guards.hpp"

namespace emel::batch::planner::modes::simple {

struct state_preparing {};
struct state_planning_input_decision {};
struct state_planning_capacity_decision {};
struct state_planning {};
struct state_planning_decision {};
struct state_planning_done {};
struct state_planning_failed {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<state_planning> <= *sml::state<state_preparing>
          + sml::event<event::plan_runtime> / action::effect_begin_planning
      , sml::state<state_planning_input_decision> <= sml::state<state_planning>
          + sml::completion<event::plan_runtime>
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_planning_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_has_invalid_step_size ]
          / action::effect_reject_invalid_step_size
      , sml::state<state_planning_capacity_decision> <= sml::state<state_planning_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_has_valid_step_size ]
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_planning_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_exceeds_step_capacity ]
          / action::effect_reject_output_steps_full
      , sml::state<state_planning_failed> <= sml::state<state_planning_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_exceeds_index_capacity ]
          / action::effect_reject_output_indices_full
      , sml::state<state_planning_decision> <= sml::state<state_planning_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_simple_plan_capacity_ok ]
          / action::effect_plan_simple_batches
      , sml::state<state_planning_done> <= sml::state<state_planning_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
      , sml::state<state_planning_failed> <= sml::state<state_planning_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed ]
          / action::effect_reject_planning_progress_stalled
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_planning_done>
          + sml::unexpected_event<sml::_>
      , sml::state<state_planning_failed> <= sml::state<state_planning_failed>
          + sml::unexpected_event<sml::_>
      , sml::state<state_planning_failed> <= sml::state<state_preparing>
          + sml::unexpected_event<sml::_>
      , sml::state<state_planning_failed> <= sml::state<state_planning>
          + sml::unexpected_event<sml::_>
      , sml::state<state_planning_failed> <= sml::state<state_planning_input_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<state_planning_failed> <= sml::state<state_planning_capacity_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<state_planning_failed> <= sml::state<state_planning_decision>
          + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

struct sm : emel::sm<model, context> {
  using base_type = emel::sm<model, context>;
  using model_type = model;
  using base_type::is;
  using base_type::visit_current_states;

  bool process_event(const event::plan_request & ev) {
    return emel::batch::planner::detail::complete_mode_request<base_type,
                                                               event::plan_request,
                                                               state_planning_done,
                                                               state_planning_failed,
                                                               events::plan_done,
                                                               events::plan_error>(
        static_cast<base_type &>(*this), ev);
  }
};

}  // namespace emel::batch::planner::modes::simple
