#pragma once
// benchmark: designed

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/transcriber/actions.hpp"
#include "emel/speech/transcriber/context.hpp"
#include "emel/speech/transcriber/events.hpp"
#include "emel/speech/transcriber/guards.hpp"

namespace emel::speech::transcriber {

struct state_uninitialized {};
struct state_initializing {};
struct state_tokenizer_decision {};
struct state_model_support_decision {};
struct state_initialize_success {};
struct state_initialize_error_out_decision {};
struct state_initialize_done_callback_decision {};
struct state_initialize_error_callback_decision {};
struct state_ready {};
struct state_recognize_support_decision {};
struct state_encoding {};
struct state_encoder_decision {};
struct state_decoding {};
struct state_decoder_decision {};
struct state_detokenizing {};
struct state_detokenize_decision {};
struct state_recognize_success {};
struct state_recognize_error_out_decision {};
struct state_recognize_done_callback_decision {};
struct state_recognize_error_callback_decision {};
struct state_recognize_uninitialized_error_out_decision {};
struct state_recognize_uninitialized_error_callback_decision {};
struct state_recognize_errored_error_out_decision {};
struct state_recognize_errored_error_callback_decision {};
struct state_done {};
struct state_errored {};

/*
speech transcriber engine (single source of truth)

state purpose
- initialize validates the injected dependencies against the request: the
  tokenizer assets must match the pinned checksum and the component contracts
  must have been bound against the same model the event carries.
- recognize drives the encode -> decode -> detokenize pipeline by dispatching
  into the injected component actors (speech/encoder, speech/decoder,
  speech/tokenizer facades); each phase outcome is an explicit decision state.
- done/error error-out and callback decision states keep the error_out and
  on_done/on_error channels as explicit transitions.
- recognize rejections dispatched before a successful initialize have their own
  origin-specific error chains (uninitialized/errored error-out and callback
  decisions) whose terminals return to the origin state, so a rejected
  pre-initialize recognize can never promote the machine to ready and open the
  pipeline without a successful initialize. The shared recognize error chain is
  reachable only from ready-origin dispatches and correctly returns to ready.

control invariants
- the machine contains no model-family names, contracts, routes, constants, or
  variant behavior; which variant runs is decided entirely by the injected
  dependencies (component kinds + contracts + decode policy), and content-level
  validation is owned by the component machines themselves.
- guards are pure predicates of (event, context); all mutation happens in
  actions; component dispatch is synchronous, acyclic, and joined before the
  action returns.
*/
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialize.
        sml::state<state_initializing> <= *sml::state<state_uninitialized>
          + sml::event<event::initialize_run> [ guard::guard_valid_initialize{} ]
          / action::effect_begin_initialize
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<event::initialize_run> [ guard::guard_invalid_initialize{} ]
          / action::effect_reject_initialize
      , sml::state<state_initializing> <= sml::state<state_ready>
          + sml::event<event::initialize_run> [ guard::guard_valid_initialize{} ]
          / action::effect_begin_initialize
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_ready>
          + sml::event<event::initialize_run> [ guard::guard_invalid_initialize{} ]
          / action::effect_reject_initialize
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_errored>
          + sml::event<event::initialize_run>
          / action::effect_reject_initialize

      , sml::state<state_tokenizer_decision> <= sml::state<state_initializing>
          + sml::completion<event::initialize_run>
      , sml::state<state_model_support_decision> <= sml::state<state_tokenizer_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_tokenizer_supported{} ]
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_tokenizer_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_tokenizer_unsupported{} ]
          / action::effect_mark_tokenizer_invalid
      , sml::state<state_initialize_success> <= sml::state<state_model_support_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_model_supported{} ]
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_model_support_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_unsupported_model{} ]
          / action::effect_mark_unsupported_model

      , sml::state<state_initialize_done_callback_decision> <= sml::state<state_initialize_success>
          + sml::completion<event::initialize_run> [ guard::guard_has_initialize_error_out{} ]
          / action::effect_store_initialize_success
      , sml::state<state_initialize_done_callback_decision> <= sml::state<state_initialize_success>
          + sml::completion<event::initialize_run> [ guard::guard_no_initialize_error_out{} ]
      , sml::state<state_initialize_error_callback_decision> <=
          sml::state<state_initialize_error_out_decision>
          + sml::completion<event::initialize_run> [ guard::guard_has_initialize_error_out{} ]
          / action::effect_store_initialize_error
      , sml::state<state_initialize_error_callback_decision> <=
          sml::state<state_initialize_error_out_decision>
          + sml::completion<event::initialize_run> [ guard::guard_no_initialize_error_out{} ]
      , sml::state<state_ready> <= sml::state<state_initialize_done_callback_decision>
          + sml::completion<event::initialize_run> [ guard::guard_has_initialize_done_callback{} ]
          / action::effect_emit_initialize_done
      , sml::state<state_ready> <= sml::state<state_initialize_done_callback_decision>
          + sml::completion<event::initialize_run> [ guard::guard_no_initialize_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_initialize_error_callback_decision>
          + sml::completion<event::initialize_run> [ guard::guard_has_initialize_error_callback{} ]
          / action::effect_emit_initialize_error
      , sml::state<state_errored> <= sml::state<state_initialize_error_callback_decision>
          + sml::completion<event::initialize_run> [ guard::guard_no_initialize_error_callback{} ]

      //------------------------------------------------------------------------------//
      // Recognition.
      , sml::state<state_recognize_support_decision> <= sml::state<state_ready>
          + sml::event<event::recognize_run> [ guard::guard_valid_recognize{} ]
          / action::effect_begin_recognize
      , sml::state<state_recognize_error_out_decision> <= sml::state<state_ready>
          + sml::event<event::recognize_run> [ guard::guard_invalid_recognize{} ]
          / action::effect_reject_recognize
      , sml::state<state_recognize_uninitialized_error_out_decision> <=
          sml::state<state_uninitialized>
          + sml::event<event::recognize_run>
          / action::effect_mark_uninitialized
      , sml::state<state_recognize_errored_error_out_decision> <= sml::state<state_errored>
          + sml::event<event::recognize_run>
          / action::effect_mark_uninitialized

      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_recognize_support_decision>
          + sml::completion<event::recognize_run>
              [ guard::guard_transcriber_unsupported{} ]
          / action::effect_mark_uninitialized
      , sml::state<state_encoding> <= sml::state<state_recognize_support_decision>
          + sml::completion<event::recognize_run>
              [ guard::guard_transcriber_ready{} ]
          / action::effect_encode
      , sml::state<state_encoder_decision> <= sml::state<state_encoding>
          + sml::completion<event::recognize_run>
      , sml::state<state_decoding> <= sml::state<state_encoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_encoder_success{} ]
          / action::effect_decode
      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_encoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_encoder_failure{} ]
          / action::effect_mark_backend_error
      , sml::state<state_decoder_decision> <= sml::state<state_decoding>
          + sml::completion<event::recognize_run>
      , sml::state<state_detokenizing> <= sml::state<state_decoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_decoder_success{} ]
          / action::effect_detokenize
      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_decoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_decoder_failure{} ]
          / action::effect_mark_backend_error
      , sml::state<state_detokenize_decision> <= sml::state<state_detokenizing>
          + sml::completion<event::recognize_run>
      , sml::state<state_recognize_success> <= sml::state<state_detokenize_decision>
          + sml::completion<event::recognize_run> [ guard::guard_detokenize_success{} ]
          / action::effect_publish_recognition_outputs
      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_detokenize_decision>
          + sml::completion<event::recognize_run> [ guard::guard_detokenize_failure{} ]
          / action::effect_mark_backend_error

      , sml::state<state_recognize_done_callback_decision> <=
          sml::state<state_recognize_success>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_out{} ]
          / action::effect_store_recognize_success
      , sml::state<state_recognize_done_callback_decision> <=
          sml::state<state_recognize_success>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_out{} ]
      , sml::state<state_recognize_error_callback_decision> <=
          sml::state<state_recognize_error_out_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_out{} ]
          / action::effect_store_recognize_error
      , sml::state<state_recognize_error_callback_decision> <=
          sml::state<state_recognize_error_out_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_out{} ]
      , sml::state<state_done> <= sml::state<state_recognize_done_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_done_callback{} ]
          / action::effect_emit_recognize_done
      , sml::state<state_done> <= sml::state<state_recognize_done_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_done_callback{} ]
      , sml::state<state_ready> <= sml::state<state_recognize_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_callback{} ]
          / action::effect_emit_recognize_error
      , sml::state<state_ready> <= sml::state<state_recognize_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::recognize_run>

      //------------------------------------------------------------------------------//
      // Pre-initialize recognize rejection (uninitialized origin). Mirrors the
      // shared recognize error chain but terminates back in
      // state_uninitialized so a rejected recognize can never promote the
      // machine to ready without a successful initialize.
      , sml::state<state_recognize_uninitialized_error_callback_decision> <=
          sml::state<state_recognize_uninitialized_error_out_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_out{} ]
          / action::effect_store_recognize_error
      , sml::state<state_recognize_uninitialized_error_callback_decision> <=
          sml::state<state_recognize_uninitialized_error_out_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_out{} ]
      , sml::state<state_uninitialized> <=
          sml::state<state_recognize_uninitialized_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_callback{} ]
          / action::effect_emit_recognize_error
      , sml::state<state_uninitialized> <=
          sml::state<state_recognize_uninitialized_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_callback{} ]

      //------------------------------------------------------------------------------//
      // Pre-initialize recognize rejection (errored origin). Same shape as the
      // uninitialized chain; terminals return to state_errored so the failed
      // initialize outcome stays observable and the pipeline stays unreachable.
      , sml::state<state_recognize_errored_error_callback_decision> <=
          sml::state<state_recognize_errored_error_out_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_out{} ]
          / action::effect_store_recognize_error
      , sml::state<state_recognize_errored_error_callback_decision> <=
          sml::state<state_recognize_errored_error_out_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_out{} ]
      , sml::state<state_errored> <=
          sml::state<state_recognize_errored_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_callback{} ]
          / action::effect_emit_recognize_error
      , sml::state<state_errored> <=
          sml::state<state_recognize_errored_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_callback{} ]

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() : sm(dependencies{}) {}

  explicit sm(const dependencies &deps)
      : base_type(), encoder_(deps.encoder_kind), decoder_(deps.decoder_kind),
        tokenizer_(deps.tokenizer_kind) {
    this->context_.deps = deps;
    this->context_.encoder = &encoder_;
    this->context_.decoder = &decoder_;
    this->context_.tokenizer = &tokenizer_;
  }

  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;
  sm(sm &&) = delete;
  sm &operator=(sm &&) = delete;

  bool process_event(const event::initialize &ev) {
    event::initialize_ctx ctx{};
    event::initialize_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == detail::to_error(error::none);
  }

  bool process_event(const event::recognize &ev) {
    event::recognize_ctx ctx{};
    event::recognize_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == detail::to_error(error::none);
  }

private:
  speech::encoder::any encoder_;
  speech::decoder::any decoder_;
  speech::tokenizer::any tokenizer_;
};

using Transcriber = sm;

} // namespace emel::speech::transcriber

namespace emel {

using SpeechTranscriber = speech::transcriber::sm;

} // namespace emel
