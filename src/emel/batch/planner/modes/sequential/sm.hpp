#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/batch/planner/modes/sequential/actions.hpp"
#include "emel/batch/planner/modes/sequential/events.hpp"
#include "emel/batch/planner/modes/sequential/guards.hpp"

namespace emel::batch::planner::modes::sequential {

struct state_preparing {};
struct state_planning {};
struct state_planning_input_decision {};
struct state_planning_capacity_decision {};
struct state_planning_execute {};
struct state_planning_result_decision {};
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
      , sml::state<state_planning_execute> <= sml::state<state_planning_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_sequential_plan_capacity_ok ]
      , sml::state<state_planning_result_decision> <= sml::state<state_planning_execute>
          + sml::completion<event::plan_runtime> / action::effect_plan_sequential_batches
      , sml::state<state_planning_done> <= sml::state<state_planning_result_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
          / action::effect_emit_plan_done
      , sml::state<state_planning_failed> <= sml::state<state_planning_result_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed ]
          / action::effect_reject_planning_progress_stalled
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_planning_done>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_failed>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_preparing>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_input_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_capacity_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_execute>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_result_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
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
    const bool accepted = base_type::process_event(ev);
    return accepted && this->is(boost::sml::state<state_planning_done>);
  }
};

}  // namespace emel::batch::planner::modes::sequential
