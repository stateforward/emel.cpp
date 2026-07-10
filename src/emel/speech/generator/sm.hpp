#pragma once

// benchmark: designed

#include <utility>

#include "emel/sm.hpp"
#include "emel/speech/generator/actions.hpp"
#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/events.hpp"
#include "emel/speech/generator/guards.hpp"

namespace emel::speech::generator {

struct state_uninitialized {};
struct state_initialize_temporal_result {};
struct state_initialize_secondary_result {};
struct state_initialize_encoder_result {};
struct state_initialize_decoder_result {};
struct state_initialize_runtime_result {};
struct state_initialize_predictor_result {};
struct state_initialize_conditioning_result {};
struct state_initialize_tokenizer_result {};
struct state_initialize_synthesis_conditioner_result {};
struct state_initialize_synthesis_prefiller_result {};
struct state_initialize_synthesis_predictor_result {};
struct state_initialize_synthesis_sampler_result {};
struct state_initialize_synthesis_decoder_result {};
struct state_initialize_synthesis_postprocessor_result {};
struct state_initialize_done_channel_decision {};
struct state_initialize_error_channel_decision {};

struct state_condition_voice {};
struct state_condition_voice_result {};
struct state_condition_prompt_begin_result {};
struct state_condition_voice_done_channel_decision {};
struct state_condition_prompt {};
struct state_condition_prompt_result {};
struct state_condition_capture_tokenizer_result {};
struct state_condition_restore_tokenizer_result {};
struct state_condition_prompt_done_channel_decision {};
struct state_condition_error_channel_decision {};

struct state_ready {};
struct state_generate_conditioning {};
struct state_generate_prefill {};
struct state_generate_predict {};
struct state_generate_sample {};
struct state_generate_decode {};
struct state_generate_postprocess {};
struct state_generate_error_channel_decision {};
struct state_generate_done_channel_decision {};

struct state_stream_encode_result {};
struct state_stream_tokenize_result {};
struct state_stream_predict_result {};
struct state_stream_detokenize_result {};
struct state_stream_decode_result {};
struct state_stream_done_channel_decision {};
struct state_stream_error_channel_decision {};

struct state_flushing {};
struct state_flush_encode_result {};
struct state_flush_tokenize_result {};
struct state_flush_predict_result {};
struct state_flush_detokenize_result {};
struct state_flush_decode_result {};
struct state_flush_done_channel_decision {};
struct state_flush_error_channel_decision {};
struct state_errored {};

/*
generic speech generator (single source of truth)

state purpose
- initialization deterministically initializes injected temporal memory,
  encoder, decoder, prediction runtime, predictor, and conditioning actors.
- condition states advance reference and prompt conditioning one bounded frame
  per public dispatch.
- state_ready accepts offline synthesis or duplex input frames.
- state_flushing advances silence frames through the same encode, predict, and
  decode actors without a hidden session loop.

control invariants
- actor and event types are compile-time injected. Generic headers contain no
  model-family contracts, handler tables, route callbacks, or runtime type
  selection.
- every child result is interpreted by guards and explicit transitions.
- child calls are synchronous, deterministically ordered, and remain inside
  the originating RTC boundary.
- request-local status stays in typed runtime events. Context owns only the
  persistent injected composition.
*/
template <class dependencies_type> struct duplex_model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    using init_run = event::initialize_run;
    using condition_run = event::condition_run;
    using generate_run = event::generate_run;
    using stream_run = event::stream_frame_run;
    using flush_run = event::flush_run;

    using init_done_present =
        guard::guard_done_callback_present<dependencies_type, init_run>;
    using init_done_absent =
        guard::guard_done_callback_absent<dependencies_type, init_run>;
    using init_error_present =
        guard::guard_error_callback_present<dependencies_type, init_run>;
    using init_error_absent =
        guard::guard_error_callback_absent<dependencies_type, init_run>;
    using condition_error_present =
        guard::guard_error_callback_present<dependencies_type, condition_run>;
    using condition_error_absent =
        guard::guard_error_callback_absent<dependencies_type, condition_run>;
    using generate_error_present =
        guard::guard_error_callback_present<dependencies_type, generate_run>;
    using generate_error_absent =
        guard::guard_error_callback_absent<dependencies_type, generate_run>;
    using stream_done_present =
        guard::guard_done_callback_present<dependencies_type, stream_run>;
    using stream_done_absent =
        guard::guard_done_callback_absent<dependencies_type, stream_run>;
    using stream_error_present =
        guard::guard_error_callback_present<dependencies_type, stream_run>;
    using stream_error_absent =
        guard::guard_error_callback_absent<dependencies_type, stream_run>;
    using flush_done_present =
        guard::guard_done_callback_present<dependencies_type, flush_run>;
    using flush_done_absent =
        guard::guard_done_callback_absent<dependencies_type, flush_run>;
    using flush_error_present =
        guard::guard_error_callback_present<dependencies_type, flush_run>;
    using flush_error_absent =
        guard::guard_error_callback_absent<dependencies_type, flush_run>;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Injected actor initialization.
        sml::state<state_initialize_temporal_result> <= *sml::state<state_uninitialized>
          + sml::event<init_run> [ guard::guard_initialize_request_valid<dependencies_type>{} ]
          / action::effect_initialize_temporal_positions<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_uninitialized> + sml::event<init_run>
          [ guard::guard_initialize_request_invalid<dependencies_type>{} ]
          / action::effect_fail_initialize<dependencies_type, error::invalid_request>{}
      , sml::state<state_initialize_secondary_result> <=
          sml::state<state_initialize_temporal_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_secondary_positions<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_temporal_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::memory_initialize_failed>{}
      , sml::state<state_initialize_encoder_result> <=
          sml::state<state_initialize_secondary_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_encoder<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_secondary_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::memory_initialize_failed>{}
      , sml::state<state_initialize_decoder_result> <=
          sml::state<state_initialize_encoder_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_decoder<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_encoder_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::encoder_initialize_failed>{}
      , sml::state<state_initialize_runtime_result> <=
          sml::state<state_initialize_decoder_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_runtime<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_decoder_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::decoder_initialize_failed>{}
      , sml::state<state_initialize_predictor_result> <=
          sml::state<state_initialize_runtime_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_predictor<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_runtime_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::runtime_initialize_failed>{}
      , sml::state<state_initialize_conditioning_result> <=
          sml::state<state_initialize_predictor_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_conditioning<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_predictor_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::predictor_initialize_failed>{}
      , sml::state<state_initialize_tokenizer_result> <=
          sml::state<state_initialize_conditioning_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_tokenizer<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_conditioning_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_initialize_done_channel_decision> <=
          sml::state<state_initialize_tokenizer_result> + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_publish_initialize_done<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_tokenizer_result> + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::tokenizer_initialize_failed>{}
      , sml::state<state_condition_voice> <=
          sml::state<state_initialize_done_channel_decision> + sml::completion<init_run>
          [ init_done_present{} ] / action::effect_emit_initialize_done<dependencies_type>{}
      , sml::state<state_condition_voice> <=
          sml::state<state_initialize_done_channel_decision> + sml::completion<init_run>
          [ init_done_absent{} ]
      , sml::state<state_errored> <=
          sml::state<state_initialize_error_channel_decision> + sml::completion<init_run>
          [ init_error_present{} ] / action::effect_emit_initialize_error<dependencies_type>{}
      , sml::state<state_errored> <=
          sml::state<state_initialize_error_channel_decision> + sml::completion<init_run>
          [ init_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Reference and prompt conditioning, one bounded frame per dispatch.
      , sml::state<state_condition_voice_result> <= sml::state<state_condition_voice>
          + sml::event<condition_run> / action::effect_condition_voice<dependencies_type>{}
      , sml::state<state_condition_voice_done_channel_decision> <=
          sml::state<state_condition_voice_result> + sml::completion<condition_run>
          [ guard::guard_condition_succeeded_pending<dependencies_type>{} ]
          / action::effect_publish_condition_pending<dependencies_type>{}
      , sml::state<state_condition_prompt_begin_result> <=
          sml::state<state_condition_voice_result> + sml::completion<condition_run>
          [ guard::guard_condition_succeeded_complete<dependencies_type>{} ]
          / action::effect_begin_prompt_conditioning<dependencies_type>{}
      , sml::state<state_condition_error_channel_decision> <=
          sml::state<state_condition_voice_result> + sml::completion<condition_run>
          [ guard::guard_condition_failed<dependencies_type>{} ]
          / action::effect_fail_condition<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_condition_voice_done_channel_decision> <=
          sml::state<state_condition_prompt_begin_result> + sml::completion<condition_run>
          [ guard::guard_child_succeeded<dependencies_type, condition_run>{} ]
          / action::effect_publish_condition_pending<dependencies_type>{}
      , sml::state<state_condition_error_channel_decision> <=
          sml::state<state_condition_prompt_begin_result> + sml::completion<condition_run>
          [ guard::guard_child_failed<dependencies_type, condition_run>{} ]
          / action::effect_fail_condition<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_condition_voice> <=
          sml::state<state_condition_voice_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_pending_done_callback_present<dependencies_type>{} ]
          / action::effect_emit_condition_done<dependencies_type>{}
      , sml::state<state_condition_voice> <=
          sml::state<state_condition_voice_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_pending_done_callback_absent<dependencies_type>{} ]
      , sml::state<state_condition_prompt> <=
          sml::state<state_condition_voice_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_complete_done_callback_present<dependencies_type>{} ]
          / action::effect_emit_condition_done<dependencies_type>{}
      , sml::state<state_condition_prompt> <=
          sml::state<state_condition_voice_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_complete_done_callback_absent<dependencies_type>{} ]
      , sml::state<state_condition_prompt_result> <= sml::state<state_condition_prompt>
          + sml::event<condition_run> / action::effect_condition_prompt<dependencies_type>{}
      , sml::state<state_condition_prompt_done_channel_decision> <=
          sml::state<state_condition_prompt_result> + sml::completion<condition_run>
          [ guard::guard_condition_succeeded_pending<dependencies_type>{} ]
          / action::effect_publish_condition_pending<dependencies_type>{}
      , sml::state<state_condition_capture_tokenizer_result> <=
          sml::state<state_condition_prompt_result> + sml::completion<condition_run>
          [ guard::guard_condition_succeeded_complete<dependencies_type>{} ]
          / action::effect_capture_tokenizer_state<dependencies_type>{}
      , sml::state<state_condition_error_channel_decision> <=
          sml::state<state_condition_prompt_result> + sml::completion<condition_run>
          [ guard::guard_condition_failed<dependencies_type>{} ]
          / action::effect_fail_condition<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_condition_restore_tokenizer_result> <=
          sml::state<state_condition_capture_tokenizer_result>
          + sml::completion<condition_run>
          [ guard::guard_child_succeeded<dependencies_type, condition_run>{} ]
          / action::effect_restore_tokenizer_state<dependencies_type>{}
      , sml::state<state_condition_error_channel_decision> <=
          sml::state<state_condition_capture_tokenizer_result>
          + sml::completion<condition_run>
          [ guard::guard_child_failed<dependencies_type, condition_run>{} ]
          / action::effect_fail_condition<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_condition_prompt_done_channel_decision> <=
          sml::state<state_condition_restore_tokenizer_result>
          + sml::completion<condition_run>
          [ guard::guard_child_succeeded<dependencies_type, condition_run>{} ]
          / action::effect_publish_condition_complete<dependencies_type>{}
      , sml::state<state_condition_error_channel_decision> <=
          sml::state<state_condition_restore_tokenizer_result>
          + sml::completion<condition_run>
          [ guard::guard_child_failed<dependencies_type, condition_run>{} ]
          / action::effect_fail_condition<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_condition_prompt> <=
          sml::state<state_condition_prompt_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_pending_done_callback_present<dependencies_type>{} ]
          / action::effect_emit_condition_done<dependencies_type>{}
      , sml::state<state_condition_prompt> <=
          sml::state<state_condition_prompt_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_pending_done_callback_absent<dependencies_type>{} ]
      , sml::state<state_ready> <=
          sml::state<state_condition_prompt_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_complete_done_callback_present<dependencies_type>{} ]
          / action::effect_emit_condition_done<dependencies_type>{}
      , sml::state<state_ready> <=
          sml::state<state_condition_prompt_done_channel_decision>
          + sml::completion<condition_run>
          [ guard::guard_condition_complete_done_callback_absent<dependencies_type>{} ]
      , sml::state<state_errored> <=
          sml::state<state_condition_error_channel_decision>
          + sml::completion<condition_run> [ condition_error_present{} ]
          / action::effect_emit_condition_error<dependencies_type>{}
      , sml::state<state_errored> <=
          sml::state<state_condition_error_channel_decision>
          + sml::completion<condition_run> [ condition_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Offline synthesis remains explicitly reserved for its later cutover.
      , sml::state<state_generate_conditioning> <= sml::state<state_ready>
          + sml::event<generate_run> [ guard::guard_generate_request_valid<dependencies_type>{} ]
          / action::effect_prepare_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <= sml::state<state_ready>
          + sml::event<generate_run> [ guard::guard_generate_request_invalid<dependencies_type>{} ]
          / action::effect_fail_generate<dependencies_type, error::invalid_request>{}
      , sml::state<state_generate_prefill> <= sml::state<state_generate_conditioning>
          + sml::completion<generate_run>
      , sml::state<state_generate_predict> <= sml::state<state_generate_prefill>
          + sml::completion<generate_run>
      , sml::state<state_generate_sample> <= sml::state<state_generate_predict>
          + sml::completion<generate_run>
      , sml::state<state_generate_decode> <= sml::state<state_generate_sample>
          + sml::completion<generate_run>
      , sml::state<state_generate_postprocess> <= sml::state<state_generate_decode>
          + sml::completion<generate_run>
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_postprocess> + sml::completion<generate_run>
          / action::effect_fail_generate<dependencies_type, error::cutover_pending>{}
      , sml::state<state_ready> <= sml::state<state_generate_error_channel_decision>
          + sml::completion<generate_run> [ generate_error_present{} ]
          / action::effect_emit_generation_error<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_generate_error_channel_decision>
          + sml::completion<generate_run> [ generate_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Duplex encode -> tokenize -> predict -> detokenize -> optional decode.
      , sml::state<state_stream_encode_result> <= sml::state<state_ready>
          + sml::event<stream_run> [ guard::guard_stream_request_valid<dependencies_type>{} ]
          / action::effect_encode_stream_frame<dependencies_type>{}
      , sml::state<state_stream_error_channel_decision> <= sml::state<state_ready>
          + sml::event<stream_run> [ guard::guard_stream_request_invalid<dependencies_type>{} ]
          / action::effect_fail_stream_frame<dependencies_type, error::invalid_request>{}
      , sml::state<state_stream_tokenize_result> <= sml::state<state_stream_encode_result>
          + sml::completion<stream_run> [ guard::guard_child_succeeded<dependencies_type, stream_run>{} ]
          / action::effect_tokenize_frame<dependencies_type, stream_run>{}
      , sml::state<state_stream_error_channel_decision> <=
          sml::state<state_stream_encode_result> + sml::completion<stream_run>
          [ guard::guard_child_failed<dependencies_type, stream_run>{} ]
          / action::effect_fail_stream_frame<dependencies_type, error::encode_failed>{}
      , sml::state<state_stream_predict_result> <= sml::state<state_stream_tokenize_result>
          + sml::completion<stream_run> [ guard::guard_child_succeeded<dependencies_type, stream_run>{} ]
          / action::effect_predict_frame<dependencies_type, stream_run>{}
      , sml::state<state_stream_error_channel_decision> <=
          sml::state<state_stream_tokenize_result> + sml::completion<stream_run>
          [ guard::guard_child_failed<dependencies_type, stream_run>{} ]
          / action::effect_fail_stream_frame<dependencies_type, error::tokenize_failed>{}
      , sml::state<state_stream_detokenize_result> <= sml::state<state_stream_predict_result>
          + sml::completion<stream_run> [ guard::guard_prediction_succeeded<dependencies_type, stream_run>{} ]
          / action::effect_detokenize_frame<dependencies_type, stream_run>{}
      , sml::state<state_stream_error_channel_decision> <=
          sml::state<state_stream_predict_result> + sml::completion<stream_run>
          [ guard::guard_prediction_failed<dependencies_type, stream_run>{} ]
          / action::effect_fail_stream_frame<dependencies_type, error::predict_failed>{}
      , sml::state<state_stream_decode_result> <= sml::state<state_stream_detokenize_result>
          + sml::completion<stream_run> [ guard::guard_frame_produced<dependencies_type, stream_run>{} ]
          / action::effect_decode_frame<dependencies_type, stream_run>{}
      , sml::state<state_stream_done_channel_decision> <=
          sml::state<state_stream_detokenize_result> + sml::completion<stream_run>
          [ guard::guard_frame_pending<dependencies_type, stream_run>{} ]
          / action::effect_publish_stream_frame_pending<dependencies_type>{}
      , sml::state<state_stream_error_channel_decision> <=
          sml::state<state_stream_detokenize_result> + sml::completion<stream_run>
          [ guard::guard_frame_failed<dependencies_type, stream_run>{} ]
          / action::effect_fail_stream_frame<dependencies_type, error::detokenize_failed>{}
      , sml::state<state_stream_done_channel_decision> <=
          sml::state<state_stream_decode_result> + sml::completion<stream_run>
          [ guard::guard_child_succeeded<dependencies_type, stream_run>{} ]
          / action::effect_publish_stream_frame_produced<dependencies_type>{}
      , sml::state<state_stream_error_channel_decision> <=
          sml::state<state_stream_decode_result> + sml::completion<stream_run>
          [ guard::guard_child_failed<dependencies_type, stream_run>{} ]
          / action::effect_fail_stream_frame<dependencies_type, error::decode_failed>{}
      , sml::state<state_ready> <= sml::state<state_stream_done_channel_decision>
          + sml::completion<stream_run> [ stream_done_present{} ]
          / action::effect_emit_stream_frame_done<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_stream_done_channel_decision>
          + sml::completion<stream_run> [ stream_done_absent{} ]
      , sml::state<state_errored> <= sml::state<state_stream_error_channel_decision>
          + sml::completion<stream_run> [ stream_error_present{} ]
          / action::effect_emit_stream_frame_error<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_stream_error_channel_decision>
          + sml::completion<stream_run> [ stream_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Silence flush uses the same actor pipeline without a hidden loop.
      , sml::state<state_flush_encode_result> <= sml::state<state_ready>
          + sml::event<flush_run> [ guard::guard_flush_request_valid<dependencies_type>{} ]
          / action::effect_encode_flush_frame<dependencies_type>{}
      , sml::state<state_flush_encode_result> <= sml::state<state_flushing>
          + sml::event<flush_run> [ guard::guard_flush_request_valid<dependencies_type>{} ]
          / action::effect_encode_flush_frame<dependencies_type>{}
      , sml::state<state_flush_error_channel_decision> <= sml::state<state_ready>
          + sml::event<flush_run> [ guard::guard_flush_request_invalid<dependencies_type>{} ]
          / action::effect_fail_flush<dependencies_type, error::invalid_request>{}
      , sml::state<state_flush_error_channel_decision> <= sml::state<state_flushing>
          + sml::event<flush_run> [ guard::guard_flush_request_invalid<dependencies_type>{} ]
          / action::effect_fail_flush<dependencies_type, error::invalid_request>{}
      , sml::state<state_flush_tokenize_result> <= sml::state<state_flush_encode_result>
          + sml::completion<flush_run> [ guard::guard_child_succeeded<dependencies_type, flush_run>{} ]
          / action::effect_tokenize_frame<dependencies_type, flush_run>{}
      , sml::state<state_flush_error_channel_decision> <=
          sml::state<state_flush_encode_result> + sml::completion<flush_run>
          [ guard::guard_child_failed<dependencies_type, flush_run>{} ]
          / action::effect_fail_flush<dependencies_type, error::encode_failed>{}
      , sml::state<state_flush_predict_result> <= sml::state<state_flush_tokenize_result>
          + sml::completion<flush_run> [ guard::guard_child_succeeded<dependencies_type, flush_run>{} ]
          / action::effect_predict_frame<dependencies_type, flush_run>{}
      , sml::state<state_flush_error_channel_decision> <=
          sml::state<state_flush_tokenize_result> + sml::completion<flush_run>
          [ guard::guard_child_failed<dependencies_type, flush_run>{} ]
          / action::effect_fail_flush<dependencies_type, error::tokenize_failed>{}
      , sml::state<state_flush_detokenize_result> <= sml::state<state_flush_predict_result>
          + sml::completion<flush_run> [ guard::guard_prediction_succeeded<dependencies_type, flush_run>{} ]
          / action::effect_detokenize_frame<dependencies_type, flush_run>{}
      , sml::state<state_flush_error_channel_decision> <=
          sml::state<state_flush_predict_result> + sml::completion<flush_run>
          [ guard::guard_prediction_failed<dependencies_type, flush_run>{} ]
          / action::effect_fail_flush<dependencies_type, error::predict_failed>{}
      , sml::state<state_flush_decode_result> <= sml::state<state_flush_detokenize_result>
          + sml::completion<flush_run> [ guard::guard_frame_produced<dependencies_type, flush_run>{} ]
          / action::effect_decode_frame<dependencies_type, flush_run>{}
      , sml::state<state_flush_done_channel_decision> <=
          sml::state<state_flush_detokenize_result> + sml::completion<flush_run>
          [ guard::guard_frame_pending<dependencies_type, flush_run>{} ]
          / action::effect_publish_flush_pending<dependencies_type>{}
      , sml::state<state_flush_error_channel_decision> <=
          sml::state<state_flush_detokenize_result> + sml::completion<flush_run>
          [ guard::guard_frame_failed<dependencies_type, flush_run>{} ]
          / action::effect_fail_flush<dependencies_type, error::detokenize_failed>{}
      , sml::state<state_flush_done_channel_decision> <=
          sml::state<state_flush_decode_result> + sml::completion<flush_run>
          [ guard::guard_child_succeeded<dependencies_type, flush_run>{} ]
          / action::effect_publish_flush_produced<dependencies_type>{}
      , sml::state<state_flush_error_channel_decision> <=
          sml::state<state_flush_decode_result> + sml::completion<flush_run>
          [ guard::guard_child_failed<dependencies_type, flush_run>{} ]
          / action::effect_fail_flush<dependencies_type, error::decode_failed>{}
      , sml::state<state_flushing> <= sml::state<state_flush_done_channel_decision>
          + sml::completion<flush_run> [ flush_done_present{} ]
          / action::effect_emit_flush_done<dependencies_type>{}
      , sml::state<state_flushing> <= sml::state<state_flush_done_channel_decision>
          + sml::completion<flush_run> [ flush_done_absent{} ]
      , sml::state<state_errored> <= sml::state<state_flush_error_channel_decision>
          + sml::completion<flush_run> [ flush_error_present{} ]
          / action::effect_emit_flush_error<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_flush_error_channel_decision>
          + sml::completion<flush_run> [ flush_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Known invalid lifecycle requests and unexpected external events.
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::uninitialized>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::cutover_pending>{}
      , sml::state<state_flushing> <= sml::state<state_flushing>
          + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::cutover_pending>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_condition_voice>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_condition_prompt>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_flushing>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
    );
    // clang-format on
  }
};

/*
generic offline synthesis composition

- the dependency mode selects this graph at compile time; there is no runtime
  route, callback table, or model-family test.
- each synthesis stage is a separately injected actor. Child acceptance and
  error results are interpreted only by guards and transitions.
- condition, stream_frame, and flush remain distinct public requests and are
  explicitly rejected by a synthesis-only composition.
*/
template <class dependencies_type> struct synthesis_model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    using init_run = event::initialize_run;
    using condition_run = event::condition_run;
    using generate_run = event::generate_run;
    using stream_run = event::stream_frame_run;
    using flush_run = event::flush_run;
    using init_done_present =
        guard::guard_done_callback_present<dependencies_type, init_run>;
    using init_done_absent =
        guard::guard_done_callback_absent<dependencies_type, init_run>;
    using init_error_present =
        guard::guard_error_callback_present<dependencies_type, init_run>;
    using init_error_absent =
        guard::guard_error_callback_absent<dependencies_type, init_run>;
    using generate_done_present =
        guard::guard_done_callback_present<dependencies_type, generate_run>;
    using generate_done_absent =
        guard::guard_done_callback_absent<dependencies_type, generate_run>;
    using generate_error_present =
        guard::guard_error_callback_present<dependencies_type, generate_run>;
    using generate_error_absent =
        guard::guard_error_callback_absent<dependencies_type, generate_run>;
    using condition_error_present =
        guard::guard_error_callback_present<dependencies_type, condition_run>;
    using condition_error_absent =
        guard::guard_error_callback_absent<dependencies_type, condition_run>;
    using stream_error_present =
        guard::guard_error_callback_present<dependencies_type, stream_run>;
    using stream_error_absent =
        guard::guard_error_callback_absent<dependencies_type, stream_run>;
    using flush_error_present =
        guard::guard_error_callback_present<dependencies_type, flush_run>;
    using flush_error_absent =
        guard::guard_error_callback_absent<dependencies_type, flush_run>;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Injected offline-synthesis actor initialization.
        sml::state<state_initialize_synthesis_conditioner_result> <=
          *sml::state<state_uninitialized> + sml::event<init_run>
          / action::effect_initialize_synthesis_conditioner<dependencies_type>{}
      , sml::state<state_initialize_synthesis_prefiller_result> <=
          sml::state<state_initialize_synthesis_conditioner_result>
          + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_synthesis_prefiller<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_synthesis_conditioner_result>
          + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::conditioner_initialize_failed>{}
      , sml::state<state_initialize_synthesis_predictor_result> <=
          sml::state<state_initialize_synthesis_prefiller_result>
          + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_synthesis_predictor<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_synthesis_prefiller_result>
          + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::prefiller_initialize_failed>{}
      , sml::state<state_initialize_synthesis_sampler_result> <=
          sml::state<state_initialize_synthesis_predictor_result>
          + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_synthesis_sampler<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_synthesis_predictor_result>
          + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::predictor_initialize_failed>{}
      , sml::state<state_initialize_synthesis_decoder_result> <=
          sml::state<state_initialize_synthesis_sampler_result>
          + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_synthesis_decoder<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_synthesis_sampler_result>
          + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::sampler_initialize_failed>{}
      , sml::state<state_initialize_synthesis_postprocessor_result> <=
          sml::state<state_initialize_synthesis_decoder_result>
          + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_initialize_synthesis_postprocessor<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_synthesis_decoder_result>
          + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::decoder_initialize_failed>{}
      , sml::state<state_initialize_done_channel_decision> <=
          sml::state<state_initialize_synthesis_postprocessor_result>
          + sml::completion<init_run>
          [ guard::guard_child_succeeded<dependencies_type, init_run>{} ]
          / action::effect_publish_initialize_done<dependencies_type>{}
      , sml::state<state_initialize_error_channel_decision> <=
          sml::state<state_initialize_synthesis_postprocessor_result>
          + sml::completion<init_run>
          [ guard::guard_child_failed<dependencies_type, init_run>{} ]
          / action::effect_fail_initialize<dependencies_type, error::postprocessor_initialize_failed>{}
      , sml::state<state_ready> <=
          sml::state<state_initialize_done_channel_decision> + sml::completion<init_run>
          [ init_done_present{} ]
          / action::effect_emit_synthesis_initialize_done<dependencies_type>{}
      , sml::state<state_ready> <=
          sml::state<state_initialize_done_channel_decision> + sml::completion<init_run>
          [ init_done_absent{} ]
      , sml::state<state_errored> <=
          sml::state<state_initialize_error_channel_decision> + sml::completion<init_run>
          [ init_error_present{} ] / action::effect_emit_initialize_error<dependencies_type>{}
      , sml::state<state_errored> <=
          sml::state<state_initialize_error_channel_decision> + sml::completion<init_run>
          [ init_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Offline condition -> prefill -> predict -> sample -> decode -> postprocess.
      , sml::state<state_generate_conditioning> <= sml::state<state_ready>
          + sml::event<generate_run> [ guard::guard_generate_request_valid<dependencies_type>{} ]
          / action::effect_condition_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <= sml::state<state_ready>
          + sml::event<generate_run> [ guard::guard_generate_request_invalid<dependencies_type>{} ]
          / action::effect_fail_generate<dependencies_type, error::invalid_request>{}
      , sml::state<state_generate_prefill> <= sml::state<state_generate_conditioning>
          + sml::completion<generate_run>
          [ guard::guard_child_succeeded<dependencies_type, generate_run>{} ]
          / action::effect_prefill_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_conditioning> + sml::completion<generate_run>
          [ guard::guard_child_failed<dependencies_type, generate_run>{} ]
          / action::effect_fail_generate<dependencies_type, error::conditioning_failed>{}
      , sml::state<state_generate_predict> <= sml::state<state_generate_prefill>
          + sml::completion<generate_run>
          [ guard::guard_child_succeeded<dependencies_type, generate_run>{} ]
          / action::effect_predict_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_prefill> + sml::completion<generate_run>
          [ guard::guard_child_failed<dependencies_type, generate_run>{} ]
          / action::effect_fail_generate<dependencies_type, error::prefill_failed>{}
      , sml::state<state_generate_sample> <= sml::state<state_generate_predict>
          + sml::completion<generate_run>
          [ guard::guard_child_succeeded<dependencies_type, generate_run>{} ]
          / action::effect_sample_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_predict> + sml::completion<generate_run>
          [ guard::guard_child_failed<dependencies_type, generate_run>{} ]
          / action::effect_fail_generate<dependencies_type, error::predict_failed>{}
      , sml::state<state_generate_decode> <= sml::state<state_generate_sample>
          + sml::completion<generate_run>
          [ guard::guard_child_succeeded<dependencies_type, generate_run>{} ]
          / action::effect_decode_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_sample> + sml::completion<generate_run>
          [ guard::guard_child_failed<dependencies_type, generate_run>{} ]
          / action::effect_fail_generate<dependencies_type, error::sample_failed>{}
      , sml::state<state_generate_postprocess> <= sml::state<state_generate_decode>
          + sml::completion<generate_run>
          [ guard::guard_child_succeeded<dependencies_type, generate_run>{} ]
          / action::effect_postprocess_generate<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_decode> + sml::completion<generate_run>
          [ guard::guard_child_failed<dependencies_type, generate_run>{} ]
          / action::effect_fail_generate<dependencies_type, error::decode_failed>{}
      , sml::state<state_generate_done_channel_decision> <=
          sml::state<state_generate_postprocess> + sml::completion<generate_run>
          [ guard::guard_child_succeeded<dependencies_type, generate_run>{} ]
          / action::effect_publish_generation_done<dependencies_type>{}
      , sml::state<state_generate_error_channel_decision> <=
          sml::state<state_generate_postprocess> + sml::completion<generate_run>
          [ guard::guard_child_failed<dependencies_type, generate_run>{} ]
          / action::effect_fail_generate<dependencies_type, error::postprocess_failed>{}
      , sml::state<state_ready> <= sml::state<state_generate_done_channel_decision>
          + sml::completion<generate_run> [ generate_done_present{} ]
          / action::effect_emit_generation_done<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_generate_done_channel_decision>
          + sml::completion<generate_run> [ generate_done_absent{} ]
      , sml::state<state_ready> <= sml::state<state_generate_error_channel_decision>
          + sml::completion<generate_run> [ generate_error_present{} ]
          / action::effect_emit_generation_error<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_generate_error_channel_decision>
          + sml::completion<generate_run> [ generate_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Public request shapes not implemented by a synthesis-only composition.
      , sml::state<state_condition_error_channel_decision> <= sml::state<state_ready>
          + sml::event<condition_run>
          / action::effect_fail_condition<dependencies_type, error::unsupported_request>{}
      , sml::state<state_ready> <= sml::state<state_condition_error_channel_decision>
          + sml::completion<condition_run> [ condition_error_present{} ]
          / action::effect_emit_condition_error<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_condition_error_channel_decision>
          + sml::completion<condition_run> [ condition_error_absent{} ]
      , sml::state<state_stream_error_channel_decision> <= sml::state<state_ready>
          + sml::event<stream_run>
          / action::effect_fail_stream_frame<dependencies_type, error::unsupported_request>{}
      , sml::state<state_ready> <= sml::state<state_stream_error_channel_decision>
          + sml::completion<stream_run> [ stream_error_present{} ]
          / action::effect_emit_stream_frame_error<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_stream_error_channel_decision>
          + sml::completion<stream_run> [ stream_error_absent{} ]
      , sml::state<state_flush_error_channel_decision> <= sml::state<state_ready>
          + sml::event<flush_run>
          / action::effect_fail_flush<dependencies_type, error::unsupported_request>{}
      , sml::state<state_ready> <= sml::state<state_flush_error_channel_decision>
          + sml::completion<flush_run> [ flush_error_present{} ]
          / action::effect_emit_flush_error<dependencies_type>{}
      , sml::state<state_ready> <= sml::state<state_flush_error_channel_decision>
          + sml::completion<flush_run> [ flush_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Lifecycle errors remain explicit.
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::uninitialized>{}
      , sml::state<state_ready> <= sml::state<state_ready> + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::cutover_pending>{}
      , sml::state<state_errored> <= sml::state<state_errored> + sml::event<event::reset>
          / action::effect_reject_reset<dependencies_type, error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
    );
    // clang-format on
  }
};

template <class dependencies_type,
          class mode_type = typename dependencies_type::generator_mode>
struct model;

template <class dependencies_type>
struct model<dependencies_type, action::mode::duplex>
    : duplex_model<dependencies_type> {};

template <class dependencies_type>
struct model<dependencies_type, action::mode::synthesis>
    : synthesis_model<dependencies_type> {};

template <class dependencies_type>
struct sm : public emel::sm<model<dependencies_type>,
                            action::context<dependencies_type>> {
  using base_type =
      emel::sm<model<dependencies_type>, action::context<dependencies_type>>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  explicit sm(const dependencies_type &deps) : base_type(std::in_place, deps) {}

  bool process_event(const event::initialize &ev) {
    event::initialize_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::initialize_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::condition &ev) {
    event::condition_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::condition_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::generate &ev) {
    event::generate_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::generate_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::stream_frame &ev) {
    event::frame_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::stream_frame_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::flush &ev) {
    event::flush_ctx ctx{};
    const bool accepted = base_type::process_event(event::flush_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::reset &ev) {
    const bool accepted = base_type::process_event(ev);
    return accepted && ev.error_out == action::error_code(error::none);
  }
};

template <class dependencies_type>
sm(const dependencies_type &) -> sm<dependencies_type>;

template <class dependencies_type> using Generator = sm<dependencies_type>;

} // namespace emel::speech::generator
