#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/diarization/request/actions.hpp"
#include "emel/diarization/request/context.hpp"
#include "emel/diarization/request/events.hpp"
#include "emel/diarization/request/guards.hpp"
#include "emel/sm.hpp"

namespace emel::diarization::request {

struct state_ready {};
struct state_model_contract_decision {};
struct state_sample_rate_decision {};
struct state_channel_count_decision {};
struct state_pcm_shape_decision {};
struct state_output_capacity_decision {};
struct state_preparing {};
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
          + sml::event<event::prepare_run>
          / action::effect_begin_prepare
      , sml::state<state_sample_rate_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::prepare_run> [ guard::guard_model_contract_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::prepare_run> [ guard::guard_model_contract_invalid{} ]
          / action::effect_mark_model_invalid

      , sml::state<state_channel_count_decision> <= sml::state<state_sample_rate_decision>
          + sml::completion<event::prepare_run> [ guard::guard_sample_rate_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_sample_rate_decision>
          + sml::completion<event::prepare_run> [ guard::guard_sample_rate_invalid{} ]
          / action::effect_mark_sample_rate_invalid

      , sml::state<state_pcm_shape_decision> <= sml::state<state_channel_count_decision>
          + sml::completion<event::prepare_run> [ guard::guard_channel_count_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_channel_count_decision>
          + sml::completion<event::prepare_run> [ guard::guard_channel_count_invalid{} ]
          / action::effect_mark_channel_count_invalid

      , sml::state<state_output_capacity_decision> <= sml::state<state_pcm_shape_decision>
          + sml::completion<event::prepare_run> [ guard::guard_pcm_shape_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_pcm_shape_decision>
          + sml::completion<event::prepare_run> [ guard::guard_pcm_shape_invalid{} ]
          / action::effect_mark_pcm_shape_invalid

      , sml::state<state_preparing> <= sml::state<state_output_capacity_decision>
          + sml::completion<event::prepare_run> [ guard::guard_output_capacity_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_output_capacity_decision>
          + sml::completion<event::prepare_run> [ guard::guard_output_capacity_invalid{} ]
          / action::effect_mark_capacity_invalid

      //------------------------------------------------------------------------------//
      // Frontend preparation.
      , sml::state<state_success_error_out_decision> <= sml::state<state_preparing>
          + sml::completion<event::prepare_run>
          / action::effect_extract_features

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::prepare_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_success_error
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::prepare_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::prepare_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_error
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::prepare_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::prepare_run> [ guard::guard_has_done_callback{} ]
          / action::effect_emit_done
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::prepare_run> [ guard::guard_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::prepare_run> [ guard::guard_has_error_callback{} ]
          / action::effect_emit_error
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::prepare_run> [ guard::guard_no_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::prepare_run>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::prepare_run>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_model_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_sample_rate_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_channel_count_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_pcm_shape_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_output_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_preparing>
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

  bool process_event(const event::prepare & ev) {
    event::prepare_ctx ctx{};
    event::prepare_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == detail::to_error(error::none);
  }
};

using Request = sm;

}  // namespace emel::diarization::request
