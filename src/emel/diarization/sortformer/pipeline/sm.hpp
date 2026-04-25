#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/diarization/sortformer/pipeline/actions.hpp"
#include "emel/diarization/sortformer/pipeline/context.hpp"
#include "emel/diarization/sortformer/pipeline/events.hpp"
#include "emel/diarization/sortformer/pipeline/guards.hpp"
#include "emel/sm.hpp"

namespace emel::diarization::sortformer::pipeline {

struct state_ready {};
struct state_model_contract_decision {};
struct state_sample_rate_decision {};
struct state_channel_count_decision {};
struct state_pcm_shape_decision {};
struct state_probability_capacity_decision {};
struct state_segment_capacity_decision {};
struct state_tensor_contract_decision {};
struct state_preparing_features {};
struct state_prepare_decision {};
struct state_binding_encoder {};
struct state_computing_encoder {};
struct state_executing_hidden {};
struct state_execute_decision {};
struct state_binding_modules {};
struct state_computing_probabilities {};
struct state_probability_decision {};
struct state_decoding_segments {};
struct state_publish_success {};
struct state_publish_error {};
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
          + sml::event<event::run_flow>
          / action::effect_begin_run
      , sml::state<state_sample_rate_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::run_flow> [ guard::guard_model_contract_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_model_contract_decision>
          + sml::completion<event::run_flow> [ guard::guard_model_contract_invalid{} ]
          / action::effect_mark_model_invalid

      , sml::state<state_channel_count_decision> <= sml::state<state_sample_rate_decision>
          + sml::completion<event::run_flow> [ guard::guard_sample_rate_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_sample_rate_decision>
          + sml::completion<event::run_flow> [ guard::guard_sample_rate_invalid{} ]
          / action::effect_mark_sample_rate_invalid

      , sml::state<state_pcm_shape_decision> <= sml::state<state_channel_count_decision>
          + sml::completion<event::run_flow> [ guard::guard_channel_count_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_channel_count_decision>
          + sml::completion<event::run_flow> [ guard::guard_channel_count_invalid{} ]
          / action::effect_mark_channel_count_invalid

      , sml::state<state_probability_capacity_decision> <= sml::state<state_pcm_shape_decision>
          + sml::completion<event::run_flow> [ guard::guard_pcm_shape_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_pcm_shape_decision>
          + sml::completion<event::run_flow> [ guard::guard_pcm_shape_invalid{} ]
          / action::effect_mark_pcm_shape_invalid

      , sml::state<state_segment_capacity_decision> <= sml::state<state_probability_capacity_decision>
          + sml::completion<event::run_flow> [ guard::guard_probability_capacity_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_probability_capacity_decision>
          + sml::completion<event::run_flow> [ guard::guard_probability_capacity_invalid{} ]
          / action::effect_mark_probability_capacity_invalid

      , sml::state<state_tensor_contract_decision> <= sml::state<state_segment_capacity_decision>
          + sml::completion<event::run_flow> [ guard::guard_segment_capacity_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_segment_capacity_decision>
          + sml::completion<event::run_flow> [ guard::guard_segment_capacity_invalid{} ]
          / action::effect_mark_segment_capacity_invalid

      , sml::state<state_preparing_features> <= sml::state<state_tensor_contract_decision>
          + sml::completion<event::run_flow> [ guard::guard_tensor_contract_valid{} ]
      , sml::state<state_publish_error> <= sml::state<state_tensor_contract_decision>
          + sml::completion<event::run_flow> [ guard::guard_tensor_contract_invalid{} ]
          / action::effect_mark_tensor_contract_invalid

      //------------------------------------------------------------------------------//
      // Maintained native path.
      , sml::state<state_prepare_decision> <= sml::state<state_preparing_features>
          + sml::completion<event::run_flow>
          / action::effect_prepare_features
      , sml::state<state_binding_encoder> <= sml::state<state_prepare_decision>
          + sml::completion<event::run_flow> [ guard::guard_no_error{} ]
      , sml::state<state_publish_error> <= sml::state<state_prepare_decision>
          + sml::completion<event::run_flow> [ guard::guard_has_error{} ]

      , sml::state<state_computing_encoder> <= sml::state<state_binding_encoder>
          + sml::completion<event::run_flow>
          / action::effect_bind_encoder
      , sml::state<state_executing_hidden> <= sml::state<state_computing_encoder>
          + sml::completion<event::run_flow>
          / action::effect_compute_encoder_frames
      , sml::state<state_execute_decision> <= sml::state<state_executing_hidden>
          + sml::completion<event::run_flow> [ guard::guard_no_error{} ]
          / action::effect_execute_hidden
      , sml::state<state_publish_error> <= sml::state<state_executing_hidden>
          + sml::completion<event::run_flow> [ guard::guard_has_error{} ]
      , sml::state<state_binding_modules> <= sml::state<state_execute_decision>
          + sml::completion<event::run_flow> [ guard::guard_no_error{} ]
      , sml::state<state_publish_error> <= sml::state<state_execute_decision>
          + sml::completion<event::run_flow> [ guard::guard_has_error{} ]

      , sml::state<state_computing_probabilities> <= sml::state<state_binding_modules>
          + sml::completion<event::run_flow>
          / action::effect_bind_modules
      , sml::state<state_probability_decision> <= sml::state<state_computing_probabilities>
          + sml::completion<event::run_flow>
          / action::effect_compute_probabilities
      , sml::state<state_decoding_segments> <= sml::state<state_probability_decision>
          + sml::completion<event::run_flow> [ guard::guard_no_error{} ]
      , sml::state<state_publish_error> <= sml::state<state_probability_decision>
          + sml::completion<event::run_flow> [ guard::guard_has_error{} ]
      , sml::state<state_publish_success> <= sml::state<state_decoding_segments>
          + sml::completion<event::run_flow>
          / action::effect_decode_segments

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_done> <= sml::state<state_publish_success>
          + sml::completion<event::run_flow>
          / action::effect_publish_success
      , sml::state<state_errored> <= sml::state<state_publish_error>
          + sml::completion<event::run_flow>
          / action::effect_publish_error
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::run_flow>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::run_flow>

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
      , sml::state<state_ready> <= sml::state<state_probability_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_segment_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_tensor_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_preparing_features>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_prepare_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_binding_encoder>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_computing_encoder>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_executing_hidden>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_execute_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_binding_modules>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_computing_probabilities>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_probability_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_decoding_segments>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_publish_success>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_publish_error>
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

  bool process_event(const event::run & ev) {
    event::run_ctx ctx{};
    event::run_flow runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == detail::to_error(error::none);
  }
};

using Pipeline = sm;

}  // namespace emel::diarization::sortformer::pipeline
