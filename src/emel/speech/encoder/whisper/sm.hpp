#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/encoder/whisper/actions.hpp"
#include "emel/speech/encoder/whisper/context.hpp"
#include "emel/speech/encoder/whisper/events.hpp"
#include "emel/speech/encoder/whisper/guards.hpp"

namespace emel::speech::encoder::whisper {

struct state_ready {};
struct state_model_contract_decision {};
struct state_sample_rate_decision {};
struct state_channel_count_decision {};
struct state_pcm_shape_decision {};
struct state_output_capacity_decision {};
struct state_workspace_capacity_decision {};
struct state_variant_decision {};
struct state_running_q8_0 {};
struct state_running_q8_0_f32_aux {};
struct state_running_q4_0 {};
struct state_running_q4_1 {};
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
          + sml::event<event::encode_run>
          / action::effect_begin_encode
      , sml::state<state_sample_rate_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::encode_run> [ guard::guard_model_contract_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::encode_run> [ guard::guard_model_contract_invalid{} ]
          / action::effect_mark_model_invalid

      , sml::state<state_channel_count_decision> <= sml::state<state_sample_rate_decision>
          + sml::completion<event::encode_run> [ guard::guard_sample_rate_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_sample_rate_decision>
          + sml::completion<event::encode_run> [ guard::guard_sample_rate_invalid{} ]
          / action::effect_mark_sample_rate_invalid

      , sml::state<state_pcm_shape_decision> <= sml::state<state_channel_count_decision>
          + sml::completion<event::encode_run> [ guard::guard_channel_count_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_channel_count_decision>
          + sml::completion<event::encode_run> [ guard::guard_channel_count_invalid{} ]
          / action::effect_mark_channel_count_invalid

      , sml::state<state_output_capacity_decision> <= sml::state<state_pcm_shape_decision>
          + sml::completion<event::encode_run> [ guard::guard_pcm_shape_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_pcm_shape_decision>
          + sml::completion<event::encode_run> [ guard::guard_pcm_shape_invalid{} ]
          / action::effect_mark_pcm_shape_invalid

      , sml::state<state_workspace_capacity_decision> <= sml::state<state_output_capacity_decision>
          + sml::completion<event::encode_run> [ guard::guard_output_capacity_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_output_capacity_decision>
          + sml::completion<event::encode_run> [ guard::guard_output_capacity_invalid{} ]
          / action::effect_mark_output_capacity_invalid

      , sml::state<state_variant_decision> <= sml::state<state_workspace_capacity_decision>
          + sml::completion<event::encode_run> [ guard::guard_workspace_capacity_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_workspace_capacity_decision>
          + sml::completion<event::encode_run> [ guard::guard_workspace_capacity_invalid{} ]
          / action::effect_mark_workspace_capacity_invalid

      //------------------------------------------------------------------------------//
      // Explicit maintained quant variant routing.
      , sml::state<state_running_q8_0_f32_aux> <= sml::state<state_variant_decision>
          + sml::completion<event::encode_run> [ guard::guard_q8_0_f32_aux_variant{} ]
          / action::effect_run_encoder_q8_0_f32_aux
      , sml::state<state_running_q8_0> <= sml::state<state_variant_decision>
          + sml::completion<event::encode_run> [ guard::guard_q8_0_variant{} ]
          / action::effect_run_encoder_q8_0
      , sml::state<state_running_q4_0> <= sml::state<state_variant_decision>
          + sml::completion<event::encode_run> [ guard::guard_q4_0_variant{} ]
          / action::effect_run_encoder_q4_0
      , sml::state<state_running_q4_1> <= sml::state<state_variant_decision>
          + sml::completion<event::encode_run> [ guard::guard_q4_1_variant{} ]
          / action::effect_run_encoder_q4_1
      , sml::state<state_error_error_out_decision> <= sml::state<state_variant_decision>
          + sml::completion<event::encode_run> [ guard::guard_unsupported_variant{} ]
          / action::effect_mark_unsupported_variant

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q8_0_f32_aux>
          + sml::completion<event::encode_run>
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q8_0>
          + sml::completion<event::encode_run>
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q4_0>
          + sml::completion<event::encode_run>
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q4_1>
          + sml::completion<event::encode_run>
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::encode_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_success_error
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::encode_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::encode_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_error
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::encode_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::encode_run> [ guard::guard_has_done_callback{} ]
          / action::effect_emit_done
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::encode_run> [ guard::guard_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::encode_run> [ guard::guard_has_error_callback{} ]
          / action::effect_emit_error
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::encode_run> [ guard::guard_no_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::encode_run>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::encode_run>

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
      , sml::state<state_ready> <= sml::state<state_workspace_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_variant_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_running_q8_0>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_running_q8_0_f32_aux>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_running_q4_0>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_running_q4_1>
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

  bool process_event(const event::encode &ev) {
    event::encode_ctx ctx{};
    event::encode_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == detail::to_error(error::none);
  }

  uint64_t q8_0_dispatch_count() const noexcept {
    return context_.q8_0_dispatch_count;
  }

  uint64_t q4_0_dispatch_count() const noexcept {
    return context_.q4_0_dispatch_count;
  }

  uint64_t q4_1_dispatch_count() const noexcept {
    return context_.q4_1_dispatch_count;
  }
};

using Encoder = sm;

} // namespace emel::speech::encoder::whisper
