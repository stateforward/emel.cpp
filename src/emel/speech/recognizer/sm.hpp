#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/recognizer/actions.hpp"
#include "emel/speech/recognizer/context.hpp"
#include "emel/speech/recognizer/events.hpp"
#include "emel/speech/recognizer/guards.hpp"

namespace emel::speech::recognizer {

struct state_uninitialized {};
struct state_initializing {};
struct state_tokenizer_decision {};
struct state_model_route_decision {};
struct state_initialize_success {};
struct state_initialize_error_out_decision {};
struct state_initialize_done_callback_decision {};
struct state_initialize_error_callback_decision {};
struct state_ready {};
struct state_recognize_route_decision {};
struct state_recognize_preparing {};
struct state_route_encoding {};
struct state_route_encoder_decision {};
struct state_route_decoding {};
struct state_route_decoder_decision {};
struct state_route_detokenizing {};
struct state_recognize_success {};
struct state_recognize_error_out_decision {};
struct state_recognize_done_callback_decision {};
struct state_recognize_error_callback_decision {};
struct state_done {};
struct state_errored {};

namespace route {

struct unsupported {
  using guard_tokenizer_supported = guard_unsupported_tokenizer;
  using guard_model_supported = guard_unsupported_model;
  using guard_recognition_ready = guard_unsupported_recognition;
  using effect_encode = effect_encode_unsupported;
  using effect_decode = effect_decode_unsupported;
  using effect_detokenize = effect_detokenize_unsupported;
};

} // namespace route

template <class route_policy = route::unsupported> struct model {
  auto operator()() const {
    namespace sml = boost::sml;

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

      , sml::state<state_tokenizer_decision> <= sml::state<state_initializing>
          + sml::completion<event::initialize_run>
      , sml::state<state_model_route_decision> <= sml::state<state_tokenizer_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_tokenizer_supported<route_policy>{} ]
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_tokenizer_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_tokenizer_unsupported<route_policy>{} ]
          / action::effect_mark_tokenizer_invalid
      , sml::state<state_initialize_success> <= sml::state<state_model_route_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_model_supported_and_route_storage_ready<route_policy>{} ]
      , sml::state<state_initialize_error_out_decision> <= sml::state<state_model_route_decision>
          + sml::completion<event::initialize_run>
              [ guard::guard_initialize_unsupported_model<route_policy>{} ]
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
      , sml::state<state_recognize_route_decision> <= sml::state<state_ready>
          + sml::event<event::recognize_run> [ guard::guard_valid_recognize{} ]
          / action::effect_begin_recognize
      , sml::state<state_recognize_error_out_decision> <= sml::state<state_ready>
          + sml::event<event::recognize_run> [ guard::guard_invalid_recognize{} ]
          / action::effect_reject_recognize
      , sml::state<state_recognize_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<event::recognize_run>
          / action::effect_mark_uninitialized
      , sml::state<state_recognize_error_out_decision> <= sml::state<state_errored>
          + sml::event<event::recognize_run>
          / action::effect_mark_uninitialized

      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_recognize_route_decision>
          + sml::completion<event::recognize_run>
              [ guard::guard_recognizer_route_unsupported<route_policy>{} ]
          / action::effect_mark_uninitialized
      , sml::state<state_route_encoding> <= sml::state<state_recognize_route_decision>
          + sml::completion<event::recognize_run>
              [ guard::guard_recognizer_route_ready<route_policy>{} ]
          / typename route_policy::effect_encode{}
      , sml::state<state_route_encoder_decision> <= sml::state<state_route_encoding>
          + sml::completion<event::recognize_run>
      , sml::state<state_route_decoding> <= sml::state<state_route_encoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_encoder_success{} ]
          / typename route_policy::effect_decode{}
      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_route_encoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_encoder_failure{} ]
          / action::effect_mark_backend_error
      , sml::state<state_route_decoder_decision> <= sml::state<state_route_decoding>
          + sml::completion<event::recognize_run>
      , sml::state<state_route_detokenizing> <= sml::state<state_route_decoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_decoder_success{} ]
          / typename route_policy::effect_detokenize{}
      , sml::state<state_recognize_error_out_decision> <=
          sml::state<state_route_decoder_decision>
          + sml::completion<event::recognize_run> [ guard::guard_decoder_failure{} ]
          / action::effect_mark_backend_error
      , sml::state<state_recognize_success> <= sml::state<state_route_detokenizing>
          + sml::completion<event::recognize_run>
          / action::effect_publish_recognition_outputs

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
      , sml::state<state_errored> <= sml::state<state_recognize_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_has_recognize_error_callback{} ]
          / action::effect_emit_recognize_error
      , sml::state<state_errored> <= sml::state<state_recognize_error_callback_decision>
          + sml::completion<event::recognize_run> [ guard::guard_no_recognize_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::recognize_run>

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

template <class route_policy = route::unsupported>
struct sm : public emel::sm<model<route_policy>, action::context> {
  using base_type = emel::sm<model<route_policy>, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;

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
};

using Recognizer = sm<>;

} // namespace emel::speech::recognizer

namespace emel {

using SpeechRecognizer = speech::recognizer::sm<>;

} // namespace emel
