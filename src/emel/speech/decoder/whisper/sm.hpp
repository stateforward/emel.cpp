#pragma once

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/decoder/whisper/actions.hpp"
#include "emel/speech/decoder/whisper/context.hpp"
#include "emel/speech/decoder/whisper/events.hpp"
#include "emel/speech/decoder/whisper/guards.hpp"

namespace emel::speech::decoder::whisper {

struct state_ready {};
struct state_model_contract_decision {};
struct state_encoder_state_decision {};
struct state_decode_policy_decision {};
struct state_generated_token_capacity_decision {};
struct state_logits_capacity_decision {};
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
          + sml::event<event::decode_run>
          / action::effect_begin_decode
      , sml::state<state_encoder_state_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::decode_run> [ guard::guard_model_contract_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_model_contract_decision>
          + sml::completion<event::decode_run> [ guard::guard_model_contract_invalid{} ]
          / action::effect_mark_model_invalid

      , sml::state<state_decode_policy_decision> <= sml::state<state_encoder_state_decision>
          + sml::completion<event::decode_run> [ guard::guard_encoder_state_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_encoder_state_decision>
          + sml::completion<event::decode_run> [ guard::guard_encoder_state_invalid{} ]
          / action::effect_mark_encoder_state_invalid

      , sml::state<state_generated_token_capacity_decision> <= sml::state<state_decode_policy_decision>
          + sml::completion<event::decode_run> [ guard::guard_decode_policy_supported{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_decode_policy_decision>
          + sml::completion<event::decode_run> [ guard::guard_decode_policy_unsupported{} ]
          / action::effect_mark_decode_policy_invalid

      , sml::state<state_logits_capacity_decision> <= sml::state<state_generated_token_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_generated_token_capacity_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_generated_token_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_generated_token_capacity_invalid{} ]
          / action::effect_mark_generated_token_capacity_invalid

      , sml::state<state_workspace_capacity_decision> <= sml::state<state_logits_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_logits_capacity_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_logits_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_logits_capacity_invalid{} ]
          / action::effect_mark_logits_capacity_invalid

      , sml::state<state_variant_decision> <= sml::state<state_workspace_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_workspace_capacity_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_workspace_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_workspace_capacity_invalid{} ]
          / action::effect_mark_workspace_capacity_invalid

      //------------------------------------------------------------------------------//
      // Explicit maintained quant variant routing.
      , sml::state<state_running_q8_0_f32_aux> <= sml::state<state_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_q8_0_f32_aux_variant{} ]
          / action::effect_run_decoder_q8_0_f32_aux
      , sml::state<state_running_q8_0> <= sml::state<state_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_q8_0_variant{} ]
          / action::effect_run_decoder_q8_0
      , sml::state<state_running_q4_0> <= sml::state<state_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_q4_0_variant{} ]
          / action::effect_run_decoder_q4_0
      , sml::state<state_running_q4_1> <= sml::state<state_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_q4_1_variant{} ]
          / action::effect_run_decoder_q4_1
      , sml::state<state_error_error_out_decision> <= sml::state<state_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_unsupported_variant{} ]
          / action::effect_mark_unsupported_variant

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q8_0_f32_aux>
          + sml::completion<event::decode_run>
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q8_0>
          + sml::completion<event::decode_run>
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q4_0>
          + sml::completion<event::decode_run>
      , sml::state<state_success_error_out_decision> <= sml::state<state_running_q4_1>
          + sml::completion<event::decode_run>
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_success_error
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_error
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_done_callback{} ]
          / action::effect_emit_done
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_error_callback{} ]
          / action::effect_emit_error
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::decode_run>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::decode_run>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_model_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_encoder_state_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_decode_policy_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_generated_token_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_logits_capacity_decision>
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

  bool process_event(const event::decode &ev) {
    event::decode_ctx ctx{};
    event::decode_run runtime_ev{ev, ctx};
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

using Decoder = sm;

} // namespace emel::speech::decoder::whisper
