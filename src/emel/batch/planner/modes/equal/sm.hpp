#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/batch/planner/modes/equal/actions.hpp"
#include "emel/batch/planner/modes/equal/events.hpp"
#include "emel/batch/planner/modes/equal/guards.hpp"

namespace emel::batch::planner::modes::equal {

struct state_preparing {};
struct state_planning {};
struct state_planning_mode_decision {};
struct state_planning_fast_input_decision {};
struct state_planning_fast_capacity_decision {};
struct state_planning_fast_execute {};
struct state_planning_general_input_decision {};
struct state_planning_general_capacity_decision {};
struct state_planning_general_execute {};
struct state_planning_general_result_decision {};
struct state_planning_fast_result_decision {};
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
      , sml::state<state_planning_mode_decision> <= sml::state<state_planning>
          + sml::completion<event::plan_runtime>
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_fast_input_decision>
          <= sml::state<state_planning_mode_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_mode_is_primary_fast_path ]
      , sml::state<state_planning_general_input_decision>
          <= sml::state<state_planning_mode_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_mode_is_general_path ]
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_general_capacity_decision>
          <= sml::state<state_planning_general_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_general_input_valid ]
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_has_invalid_step_size ]
          / action::effect_reject_invalid_step_size
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_lacks_step_capacity ]
          / action::effect_reject_output_steps_full
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_lacks_index_capacity ]
          / action::effect_reject_output_indices_full
      , sml::state<state_planning_general_execute>
          <= sml::state<state_planning_general_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_storage_capacity_valid ]
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_general_result_decision>
          <= sml::state<state_planning_general_execute>
          + sml::completion<event::plan_runtime> / action::effect_plan_equal_batches
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_has_invalid_step_size ]
          / action::effect_reject_invalid_step_size
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_fast_path_missing_primary_ids ]
          / action::effect_reject_invalid_sequence_id
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_fast_path_primary_ids_invalid ]
          / action::effect_reject_invalid_sequence_id
      , sml::state<state_planning_fast_capacity_decision>
          <= sml::state<state_planning_fast_input_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_fast_path_input_valid ]
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_lacks_step_capacity ]
          / action::effect_reject_output_steps_full
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_lacks_index_capacity ]
          / action::effect_reject_output_indices_full
      , sml::state<state_planning_fast_execute>
          <= sml::state<state_planning_fast_capacity_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_storage_capacity_valid ]
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_fast_result_decision>
          <= sml::state<state_planning_fast_execute>
          + sml::completion<event::plan_runtime> / action::effect_plan_equal_primary_batches
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_done> <= sml::state<state_planning_general_result_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
          / action::effect_emit_plan_done
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_result_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed ]
          / action::effect_reject_planning_progress_stalled
      , sml::state<state_planning_done> <= sml::state<state_planning_fast_result_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
          / action::effect_emit_plan_done
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_result_decision>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed ]
          / action::effect_reject_planning_progress_stalled
      //------------------------------------------------------------------------------//
      , sml::state<state_planning_failed> <= sml::state<state_preparing>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_mode_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_input_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_capacity_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_execute>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_input_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_capacity_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_execute>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_general_result_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_fast_result_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_done>
          + sml::unexpected_event<sml::_>
          / action::effect_emit_internal_plan_error
      , sml::state<state_planning_failed> <= sml::state<state_planning_failed>
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

}  // namespace emel::batch::planner::modes::equal
