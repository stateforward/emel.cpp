#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/batch/planner/modes/equal/actions.hpp"
#include "emel/batch/planner/modes/equal/guards.hpp"

namespace emel::batch::planner::modes::equal {

struct preparing {};
struct planning {};
struct planning_mode_decision {};
struct planning_fast_input_decision {};
struct planning_fast_capacity_decision {};
struct planning_fast_group_scan_decision {};
struct planning_fast_progress_decision {};
struct planning_fast_execute {};
struct planning_general_input_decision {};
struct planning_general_capacity_decision {};
struct planning_general_group_scan_decision {};
struct planning_general_progress_decision {};
struct planning_general_execute {};
struct planning_general_result_decision {};
struct planning_fast_result_decision {};
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
      , sml::state<planning_fast_input_decision> <= sml::state<planning_mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_primary_fast_path ]
      , sml::state<planning_general_input_decision> <= sml::state<planning_mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_general_path ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_general_capacity_decision> <= sml::state<planning_general_input_decision>
          + sml::completion<event::request_runtime> [ guard::general_input_valid ]
      , sml::state<planning_failed> <= sml::state<planning_general_input_decision>
          + sml::completion<event::request_runtime> [ guard::has_invalid_step_size ]
          / action::mark_invalid_step_size
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_general_capacity_decision>
          + sml::completion<event::request_runtime> [ guard::lacks_step_capacity ]
          / action::mark_output_steps_full
      , sml::state<planning_failed> <= sml::state<planning_general_capacity_decision>
          + sml::completion<event::request_runtime> [ guard::lacks_index_capacity ]
          / action::mark_output_indices_full
      , sml::state<planning_general_group_scan_decision> <= sml::state<planning_general_capacity_decision>
          + sml::completion<event::request_runtime> [ guard::storage_capacity_valid ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_general_group_scan_decision>
          + sml::completion<event::request_runtime> [ guard::general_first_group_scan_exceeds_step_size ]
          / action::mark_planning_progress_stalled
      , sml::state<planning_general_progress_decision> <= sml::state<planning_general_group_scan_decision>
          + sml::completion<event::request_runtime> [ guard::general_first_group_scan_within_step_size ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_general_progress_decision>
          + sml::completion<event::request_runtime> [ guard::general_progress_not_modelable ]
          / action::mark_planning_progress_stalled
      , sml::state<planning_general_execute> <= sml::state<planning_general_progress_decision>
          + sml::completion<event::request_runtime> [ guard::general_progress_modelable ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_general_result_decision> <= sml::state<planning_general_execute>
          + sml::completion<event::request_runtime> / action::create_plan_general
      , sml::state<planning_failed> <= sml::state<planning_fast_input_decision>
          + sml::completion<event::request_runtime> [ guard::has_invalid_step_size ]
          / action::mark_invalid_step_size
      , sml::state<planning_failed> <= sml::state<planning_fast_input_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_missing_primary_ids ]
          / action::mark_invalid_sequence_id
      , sml::state<planning_failed> <= sml::state<planning_fast_input_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_primary_ids_invalid ]
          / action::mark_invalid_sequence_id
      , sml::state<planning_fast_capacity_decision> <= sml::state<planning_fast_input_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_input_valid ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_fast_capacity_decision>
          + sml::completion<event::request_runtime> [ guard::lacks_step_capacity ]
          / action::mark_output_steps_full
      , sml::state<planning_failed> <= sml::state<planning_fast_capacity_decision>
          + sml::completion<event::request_runtime> [ guard::lacks_index_capacity ]
          / action::mark_output_indices_full
      , sml::state<planning_fast_group_scan_decision> <= sml::state<planning_fast_capacity_decision>
          + sml::completion<event::request_runtime> [ guard::storage_capacity_valid ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_fast_group_scan_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_first_group_scan_exceeds_step_size ]
          / action::mark_planning_progress_stalled
      , sml::state<planning_fast_progress_decision> <= sml::state<planning_fast_group_scan_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_first_group_scan_within_step_size ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_failed> <= sml::state<planning_fast_progress_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_progress_not_modelable ]
          / action::mark_planning_progress_stalled
      , sml::state<planning_fast_execute> <= sml::state<planning_fast_progress_decision>
          + sml::completion<event::request_runtime> [ guard::fast_path_progress_modelable ]
      //------------------------------------------------------------------------------//
      , sml::state<planning_fast_result_decision> <= sml::state<planning_fast_execute>
          + sml::completion<event::request_runtime> / action::create_plan_primary_fast_path
      //------------------------------------------------------------------------------//
      , sml::state<planning_done> <= sml::state<planning_general_result_decision>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ]
      , sml::state<planning_failed> <= sml::state<planning_general_result_decision>
          + sml::completion<event::request_runtime> [ guard::planning_failed ]
      , sml::state<planning_done> <= sml::state<planning_fast_result_decision>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ]
      , sml::state<planning_failed> <= sml::state<planning_fast_result_decision>
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
      , sml::state<planning_failed> <= sml::state<planning_fast_input_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_fast_capacity_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_fast_group_scan_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_fast_progress_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_fast_execute>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general_input_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general_capacity_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general_group_scan_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general_progress_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general_execute>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_general_result_decision>
          + sml::unexpected_event<sml::_>
      , sml::state<planning_failed> <= sml::state<planning_fast_result_decision>
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
