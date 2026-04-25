#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/diarization/sortformer/executor/actions.hpp"
#include "emel/diarization/sortformer/executor/context.hpp"
#include "emel/diarization/sortformer/executor/events.hpp"
#include "emel/diarization/sortformer/executor/guards.hpp"
#include "emel/sm.hpp"

namespace emel::diarization::sortformer::executor {

struct state_ready {};
struct state_model_contract_decision {};
struct state_tensor_contract_decision {};
struct state_input_shape_decision {};
struct state_output_capacity_decision {};
struct state_binding {};
struct state_projecting {};
struct state_transformer_cache {};
struct state_transformer_layer_00 {};
struct state_transformer_layer_01 {};
struct state_transformer_layer_02 {};
struct state_transformer_layer_03 {};
struct state_transformer_layer_04 {};
struct state_transformer_layer_05 {};
struct state_transformer_layer_06 {};
struct state_transformer_layer_07 {};
struct state_transformer_layer_08 {};
struct state_transformer_layer_09 {};
struct state_transformer_layer_10 {};
struct state_transformer_layer_11 {};
struct state_transformer_layer_12 {};
struct state_transformer_layer_13 {};
struct state_transformer_layer_14 {};
struct state_transformer_layer_15 {};
struct state_transformer_layer_16 {};
struct state_transformer_layer_17 {};
struct state_publishing_hidden {};
struct state_success_error_out_decision {};
struct state_success_callback_decision {};
struct state_error_error_out_decision {};
struct state_error_callback_decision {};
struct state_done {};
struct state_errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<state_model_contract_decision> <= *sml::state<state_ready>
          + sml::event<event::execute_run>
          / action::effect_begin_execute
      , sml::state<state_tensor_contract_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::execute_run> [ guard::guard_model_contract_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::execute_run> [ guard::guard_model_contract_invalid{} ]
          / action::effect_mark_model_invalid

      , sml::state<state_input_shape_decision> <= sml::state<state_tensor_contract_decision>
          + sml::completion<event::execute_run> [ guard::guard_tensor_contract_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_tensor_contract_decision>
          + sml::completion<event::execute_run> [ guard::guard_tensor_contract_invalid{} ]
          / action::effect_mark_tensor_contract_invalid

      , sml::state<state_output_capacity_decision> <= sml::state<state_input_shape_decision>
          + sml::completion<event::execute_run> [ guard::guard_input_shape_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_input_shape_decision>
          + sml::completion<event::execute_run> [ guard::guard_input_shape_invalid{} ]
          / action::effect_mark_input_shape_invalid

      , sml::state<state_binding> <= sml::state<state_output_capacity_decision>
          + sml::completion<event::execute_run> [ guard::guard_output_capacity_valid{} ]
          / action::effect_bind_contracts
      , sml::state<state_error_error_out_decision> <= sml::state<state_output_capacity_decision>
          + sml::completion<event::execute_run> [ guard::guard_output_capacity_invalid{} ]
          / action::effect_mark_output_capacity_invalid

      //------------------------------------------------------------------------------//
      // Stage execution.
      , sml::state<state_projecting> <= sml::state<state_binding>
          + sml::completion<event::execute_run>
          / action::effect_project_encoder
      , sml::state<state_transformer_cache> <= sml::state<state_projecting>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_write_projected_frames_to_cache
      , sml::state<state_error_error_out_decision> <= sml::state<state_projecting>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_00> <= sml::state<state_transformer_cache>
          + sml::completion<event::execute_run>
          / action::effect_execute_transformer_layer_00
      , sml::state<state_transformer_layer_01> <= sml::state<state_transformer_layer_00>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_01
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_00>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_02> <= sml::state<state_transformer_layer_01>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_02
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_01>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_03> <= sml::state<state_transformer_layer_02>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_03
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_02>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_04> <= sml::state<state_transformer_layer_03>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_04
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_03>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_05> <= sml::state<state_transformer_layer_04>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_05
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_04>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_06> <= sml::state<state_transformer_layer_05>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_06
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_05>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_07> <= sml::state<state_transformer_layer_06>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_07
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_06>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_08> <= sml::state<state_transformer_layer_07>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_08
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_07>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_09> <= sml::state<state_transformer_layer_08>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_09
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_08>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_10> <= sml::state<state_transformer_layer_09>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_10
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_09>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_11> <= sml::state<state_transformer_layer_10>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_11
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_10>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_12> <= sml::state<state_transformer_layer_11>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_12
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_11>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_13> <= sml::state<state_transformer_layer_12>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_13
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_12>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_14> <= sml::state<state_transformer_layer_13>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_14
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_13>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_15> <= sml::state<state_transformer_layer_14>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_15
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_14>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_16> <= sml::state<state_transformer_layer_15>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_16
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_15>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_transformer_layer_17> <= sml::state<state_transformer_layer_16>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_execute_transformer_layer_17
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_16>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_publishing_hidden> <= sml::state<state_transformer_layer_17>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
          / action::effect_publish_hidden
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_layer_17>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]
      , sml::state<state_success_error_out_decision> <= sml::state<state_publishing_hidden>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::execute_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_success_error
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::execute_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::execute_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_error
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::execute_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::execute_run> [ guard::guard_has_done_callback{} ]
          / action::effect_emit_done
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::execute_run> [ guard::guard_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::execute_run> [ guard::guard_has_error_callback{} ]
          / action::effect_emit_error
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::execute_run> [ guard::guard_no_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::execute_run>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::execute_run>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_model_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_tensor_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_input_shape_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_output_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_binding>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_projecting>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_cache>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_00>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_01>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_02>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_03>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_04>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_05>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_06>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_07>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_08>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_09>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_10>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_11>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_12>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_13>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_14>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_15>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_16>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_transformer_layer_17>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_publishing_hidden>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_success_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_success_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_error_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_error_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;

  bool process_event(const event::execute & ev) {
    event::execute_ctx ctx{};
    event::execute_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == detail::to_error(error::none);
  }
};

using Executor = sm;

}  // namespace emel::diarization::sortformer::executor
