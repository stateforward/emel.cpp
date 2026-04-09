#pragma once

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/sm.hpp"

namespace emel::batch::planner {

struct state_idle {};
struct state_input_validation {};
struct state_step_normalization {};
struct state_mode_selection {};
struct state_simple_planning {};
struct state_equal_planning {};
struct state_sequential_planning {};
struct state_result_publish {};
struct state_completed {};
struct state_request_rejected {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<state_input_validation> <= *sml::state<state_idle>
          + sml::event<event::plan_runtime>
          / action::effect_begin_planning
      , sml::state<state_step_normalization> <= sml::state<state_input_validation>
          + sml::completion<event::plan_runtime> [ guard::guard_inputs_valid ]
      , sml::state<state_request_rejected> <= sml::state<state_input_validation>
          + sml::completion<event::plan_runtime> [ guard::guard_inputs_invalid ]
          / action::effect_reject_invalid_request
      //------------------------------------------------------------------------------//
      , sml::state<state_mode_selection> <= sml::state<state_step_normalization>
          + sml::completion<event::plan_runtime> / action::effect_normalize_step_size
      //------------------------------------------------------------------------------//
      , sml::state<state_simple_planning> <= sml::state<state_mode_selection>
          + sml::completion<event::plan_runtime> [ guard::guard_mode_is_simple ]
          / action::effect_plan_simple_mode
      , sml::state<state_equal_planning> <= sml::state<state_mode_selection>
          + sml::completion<event::plan_runtime> [ guard::guard_mode_is_equal ]
          / action::effect_plan_equal_mode
      , sml::state<state_sequential_planning> <= sml::state<state_mode_selection>
          + sml::completion<event::plan_runtime> [ guard::guard_mode_is_sequential ]
          / action::effect_plan_sequential_mode
      , sml::state<state_request_rejected> <= sml::state<state_mode_selection>
          + sml::completion<event::plan_runtime> [ guard::guard_mode_is_invalid ]
          / action::effect_reject_invalid_mode
      //------------------------------------------------------------------------------//
      , sml::state<state_result_publish> <= sml::state<state_simple_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
          / action::effect_publish_result
      , sml::state<state_completed> <= sml::state<state_simple_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed_with_error ]
          / action::effect_emit_planning_error
      , sml::state<state_completed> <= sml::state<state_simple_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed_without_error ]
          / action::effect_emit_internal_planning_error
      , sml::state<state_result_publish> <= sml::state<state_equal_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
          / action::effect_publish_result
      , sml::state<state_completed> <= sml::state<state_equal_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed_with_error ]
          / action::effect_emit_planning_error
      , sml::state<state_completed> <= sml::state<state_equal_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed_without_error ]
          / action::effect_emit_internal_planning_error
      , sml::state<state_result_publish> <= sml::state<state_sequential_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_succeeded ]
          / action::effect_publish_result
      , sml::state<state_completed> <= sml::state<state_sequential_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed_with_error ]
          / action::effect_emit_planning_error
      , sml::state<state_completed> <= sml::state<state_sequential_planning>
          + sml::completion<event::plan_runtime> [ guard::guard_planning_failed_without_error ]
          / action::effect_emit_internal_planning_error
      //------------------------------------------------------------------------------//
      , sml::state<state_completed> <= sml::state<state_result_publish>
          + sml::completion<event::plan_runtime> / action::effect_emit_plan_done
      //------------------------------------------------------------------------------//
      , sml::state<state_input_validation> <= sml::state<state_completed>
          + sml::event<event::plan_runtime>
          / action::effect_begin_planning
      , sml::state<state_input_validation> <= sml::state<state_request_rejected>
          + sml::event<event::plan_runtime> / action::effect_begin_planning
      //------------------------------------------------------------------------------//
      , sml::state<state_idle> <= sml::state<state_idle>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_input_validation>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_step_normalization>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_mode_selection>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_simple_planning>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_equal_planning>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_sequential_planning>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_result_publish>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_completed>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
      , sml::state<state_idle> <= sml::state<state_request_rejected>
          + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected_event
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  bool process_event(const event::plan_request & ev) {
    event::plan_scratch ctx{};
    event::plan_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Planner = sm;

}  // namespace emel::batch::planner

namespace emel {

using BatchPlanner = batch::planner::Planner;

}  // namespace emel
