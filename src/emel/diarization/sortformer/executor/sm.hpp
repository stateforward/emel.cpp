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
struct state_executing {};
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
      , sml::state<state_executing> <= sml::state<state_binding>
          + sml::completion<event::execute_run>
          / action::effect_execute_stage
      , sml::state<state_success_error_out_decision> <= sml::state<state_executing>
          + sml::completion<event::execute_run> [ guard::guard_execution_ok{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_executing>
          + sml::completion<event::execute_run> [ guard::guard_execution_failed{} ]

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
      , sml::state<state_ready> <= sml::state<state_executing>
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
