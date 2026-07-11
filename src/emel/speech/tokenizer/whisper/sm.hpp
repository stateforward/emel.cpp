#pragma once
// benchmark: designed

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/tokenizer/whisper/actions.hpp"
#include "emel/speech/tokenizer/whisper/events.hpp"
#include "emel/speech/tokenizer/whisper/guards.hpp"

namespace emel::speech::tokenizer::whisper {

struct state_ready {};
struct state_json_decision {};
struct state_detokenizing {};
struct state_success_error_out_decision {};
struct state_success_callback_decision {};
struct state_error_error_out_decision {};
struct state_error_callback_decision {};
struct state_done {};
struct state_errored {};
struct state_validate_decision {};
struct state_validate_success_error_out_decision {};
struct state_validate_error_error_out_decision {};
struct state_validate_done {};
struct state_validate_errored {};

/*
whisper detokenize machine

state purpose
- state_json_decision validates the request before any transcript bytes are
  written: the tokenizer JSON must carry the tiny control tokens, the
  caller-owned spans must be well formed, and every token ID must be
  non-negative (vocab indices; negative IDs would reach the signed-magnitude
  rendering in the decode path).
- state_detokenizing performs the bounded token-id -> text decode in a single
  transition (data-plane iteration only; no behavior selection). The decode
  truncates to the caller's transcript capacity, matching the public
  transcriber contract.
- success/error error-out and callback decision states mirror the encoder and
  decoder machines so error_out and on_done/on_error channels are explicit
  transitions, never action-side branching.
- state_done / state_errored return to state_ready by completion, so one
  machine instance serves repeated detokenize dispatches.
- the validate flow (state_validate_*) serves owner-driven asset validation:
  it checks the tokenizer JSON control tokens and the bound decode policy so
  owners (e.g. the transcriber's initialize phase) can fail fast instead of
  deferring the failure to the first detokenize after pipeline work ran. The
  flow publishes only through error_out and returns to state_ready.

invariants
- guards are pure predicates of the dispatch-run event; all context mutation
  happens in actions.
- the machine holds no persistent context; every dispatch is self-contained.
*/
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<state_json_decision> <= *sml::state<state_ready>
          + sml::event<event::detokenize_run>
          / action::effect_begin_detokenize{}
      , sml::state<state_detokenizing> <= sml::state<state_json_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_detokenize_request_valid{} ]
          / action::effect_detokenize{}
      , sml::state<state_error_error_out_decision> <= sml::state<state_json_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_tokenizer_json_invalid{} ]
          / action::effect_mark_tokenizer_json_invalid{}
      , sml::state<state_error_error_out_decision> <= sml::state<state_json_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_token_ids_invalid{} ]
          / action::effect_mark_token_ids_invalid{}

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_success_error_out_decision> <= sml::state<state_detokenizing>
          + sml::completion<event::detokenize_run>
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_out{}
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_out{}
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_has_done_callback{} ]
          / action::effect_emit_done{}
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_has_error_callback{} ]
          / action::effect_emit_error{}
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::detokenize_run> [ guard::guard_no_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::detokenize_run>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::detokenize_run>

      //------------------------------------------------------------------------------//
      // Asset validation (owner-driven, error_out-only publication).
      , sml::state<state_validate_decision> <= sml::state<state_ready>
          + sml::event<event::validate_run>
          / action::effect_begin_validate{}
      , sml::state<state_validate_success_error_out_decision> <=
          sml::state<state_validate_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_supported{} ]
      , sml::state<state_validate_error_error_out_decision> <=
          sml::state<state_validate_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_json_invalid{} ]
          / action::effect_mark_validate_json_invalid{}
      , sml::state<state_validate_error_error_out_decision> <=
          sml::state<state_validate_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_policy_unsupported{} ]
          / action::effect_mark_validate_policy_unsupported{}
      , sml::state<state_validate_done> <= sml::state<state_validate_success_error_out_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_has_error_out{} ]
          / action::effect_store_validate_error_out{}
      , sml::state<state_validate_done> <= sml::state<state_validate_success_error_out_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_no_error_out{} ]
      , sml::state<state_validate_errored> <= sml::state<state_validate_error_error_out_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_has_error_out{} ]
          / action::effect_store_validate_error_out{}
      , sml::state<state_validate_errored> <= sml::state<state_validate_error_error_out_decision>
          + sml::completion<event::validate_run> [ guard::guard_validate_no_error_out{} ]
      , sml::state<state_ready> <= sml::state<state_validate_done>
          + sml::completion<event::validate_run>
      , sml::state<state_ready> <= sml::state<state_validate_errored>
          + sml::completion<event::validate_run>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_json_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_detokenizing>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_success_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_success_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_error_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_error_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_validate_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_validate_success_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_validate_error_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_validate_done>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_validate_errored>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;

  bool process_event(const event::detokenize &ev) {
    event::detokenize_ctx ctx{};
    event::detokenize_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::validate &ev) {
    event::validate_ctx ctx{};
    event::validate_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

} // namespace emel::speech::tokenizer::whisper
