#pragma once
// benchmark: designed

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/codec/mimi/actions.hpp"
#include "emel/speech/codec/mimi/context.hpp"
#include "emel/speech/codec/mimi/events.hpp"
#include "emel/speech/codec/mimi/guards.hpp"

// Mimi codec facade: owns the bound runtime, the streaming state, and the
// frontend/quantizer/backend child actors, and sequences one 80 ms frame per
// dispatch through them. Every success/error route is decided by pure
// validation guards BEFORE the corresponding action runs (bind contract via
// the dry-run walk, arena capacity via the sizing contract, frame requests
// against the bound runtime), so the compute actions are non-failing and no
// action-written flag routes behavior. Requests before initialization are
// answered with explicit not_initialized errors rather than dropped.
namespace emel::speech::codec::mimi {

struct state_uninitialized {};
struct state_bind_contract_decision {};
struct state_bind_capacity_decision {};
struct state_binding {};
struct state_init_error_out_decision {};
struct state_init_callback_decision {};
struct state_init_failed_error_out_decision {};
struct state_init_failed_callback_decision {};
struct state_session_ready {};
struct state_encode_request_decision {};
struct state_encoding {};
struct state_quantizing {};
struct state_encode_error_out_decision {};
struct state_encode_callback_decision {};
struct state_encode_failed_error_out_decision {};
struct state_encode_failed_callback_decision {};
struct state_decode_request_decision {};
struct state_decode_codes_decision {};
struct state_dequantizing {};
struct state_decoding {};
struct state_decode_error_out_decision {};
struct state_decode_callback_decision {};
struct state_decode_failed_error_out_decision {};
struct state_decode_failed_callback_decision {};
struct state_uninit_encode_error_out_decision {};
struct state_uninit_encode_callback_decision {};
struct state_uninit_decode_error_out_decision {};
struct state_uninit_decode_callback_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using init_run = event::initialize_run;
    using encode_run = event::encode_frame_run;
    using decode_run = event::decode_frame_run;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialization: contract, then capacity, then the non-failing bind.
        sml::state<state_bind_contract_decision> <= *sml::state<state_uninitialized>
          + sml::event<init_run>
      , sml::state<state_bind_capacity_decision> <= sml::state<state_bind_contract_decision>
          + sml::completion<init_run> [ guard::guard_bind_contract_valid{} ]
      , sml::state<state_init_failed_error_out_decision> <= sml::state<state_bind_contract_decision>
          + sml::completion<init_run> [ guard::guard_bind_contract_invalid{} ]
          / action::effect_mark_bind_failed{}
      , sml::state<state_binding> <= sml::state<state_bind_capacity_decision>
          + sml::completion<init_run> [ guard::guard_arena_capacity_valid{} ]
          / action::effect_bind{}
      , sml::state<state_init_failed_error_out_decision> <= sml::state<state_bind_capacity_decision>
          + sml::completion<init_run> [ guard::guard_arena_capacity_invalid{} ]
          / action::effect_mark_arena_capacity_invalid{}
      , sml::state<state_init_error_out_decision> <= sml::state<state_binding>
          + sml::completion<init_run>
      , sml::state<state_init_callback_decision> <= sml::state<state_init_error_out_decision>
          + sml::completion<init_run> [ guard::guard_has_error_out<init_run>{} ]
          / action::effect_store_error_out<init_run>{}
      , sml::state<state_init_callback_decision> <= sml::state<state_init_error_out_decision>
          + sml::completion<init_run> [ guard::guard_no_error_out<init_run>{} ]
      , sml::state<state_init_failed_callback_decision> <= sml::state<state_init_failed_error_out_decision>
          + sml::completion<init_run> [ guard::guard_has_error_out<init_run>{} ]
          / action::effect_store_error_out<init_run>{}
      , sml::state<state_init_failed_callback_decision> <= sml::state<state_init_failed_error_out_decision>
          + sml::completion<init_run> [ guard::guard_no_error_out<init_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_init_callback_decision>
          + sml::completion<init_run> [ guard::guard_has_done_callback<init_run>{} ]
          / action::effect_emit_initialize_done{}
      , sml::state<state_session_ready> <= sml::state<state_init_callback_decision>
          + sml::completion<init_run> [ guard::guard_no_done_callback<init_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_callback_decision>
          + sml::completion<init_run> [ guard::guard_has_error_callback<init_run>{} ]
          / action::effect_emit_initialize_error{}
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_callback_decision>
          + sml::completion<init_run> [ guard::guard_no_error_callback<init_run>{} ]

      //------------------------------------------------------------------------------//
      // Frame encode (session): request validation, then non-failing stages.
      , sml::state<state_encode_request_decision> <= sml::state<state_session_ready>
          + sml::event<encode_run>
      , sml::state<state_encoding> <= sml::state<state_encode_request_decision>
          + sml::completion<encode_run> [ guard::guard_encode_request_valid{} ]
          / action::effect_run_frontend_child{}
      , sml::state<state_encode_failed_error_out_decision> <= sml::state<state_encode_request_decision>
          + sml::completion<encode_run> [ guard::guard_encode_request_invalid{} ]
          / action::effect_mark_request_shape_invalid<encode_run>{}
      , sml::state<state_quantizing> <= sml::state<state_encoding>
          + sml::completion<encode_run>
          / action::effect_run_quantize_child{}
      , sml::state<state_encode_error_out_decision> <= sml::state<state_quantizing>
          + sml::completion<encode_run>
      , sml::state<state_encode_callback_decision> <= sml::state<state_encode_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_out<encode_run>{} ]
          / action::effect_store_error_out<encode_run>{}
      , sml::state<state_encode_callback_decision> <= sml::state<state_encode_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_out<encode_run>{} ]
      , sml::state<state_encode_failed_callback_decision> <= sml::state<state_encode_failed_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_out<encode_run>{} ]
          / action::effect_store_error_out<encode_run>{}
      , sml::state<state_encode_failed_callback_decision> <= sml::state<state_encode_failed_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_out<encode_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_encode_callback_decision>
          + sml::completion<encode_run> [ guard::guard_has_done_callback<encode_run>{} ]
          / action::effect_emit_encode_done{}
      , sml::state<state_session_ready> <= sml::state<state_encode_callback_decision>
          + sml::completion<encode_run> [ guard::guard_no_done_callback<encode_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_encode_failed_callback_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_callback<encode_run>{} ]
          / action::effect_emit_encode_error{}
      , sml::state<state_session_ready> <= sml::state<state_encode_failed_callback_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_callback<encode_run>{} ]

      //------------------------------------------------------------------------------//
      // Frame decode (session): request shape, then code range, then the
      // non-failing stages.
      , sml::state<state_decode_request_decision> <= sml::state<state_session_ready>
          + sml::event<decode_run>
      , sml::state<state_decode_codes_decision> <= sml::state<state_decode_request_decision>
          + sml::completion<decode_run> [ guard::guard_decode_request_valid{} ]
      , sml::state<state_decode_failed_error_out_decision> <= sml::state<state_decode_request_decision>
          + sml::completion<decode_run> [ guard::guard_decode_request_invalid{} ]
          / action::effect_mark_request_shape_invalid<decode_run>{}
      , sml::state<state_dequantizing> <= sml::state<state_decode_codes_decision>
          + sml::completion<decode_run> [ guard::guard_decode_codes_valid{} ]
          / action::effect_run_dequantize_child{}
      , sml::state<state_decode_failed_error_out_decision> <= sml::state<state_decode_codes_decision>
          + sml::completion<decode_run> [ guard::guard_decode_codes_invalid{} ]
          / action::effect_mark_code_range_invalid{}
      , sml::state<state_decoding> <= sml::state<state_dequantizing>
          + sml::completion<decode_run>
          / action::effect_run_backend_child{}
      , sml::state<state_decode_error_out_decision> <= sml::state<state_decoding>
          + sml::completion<decode_run>
      , sml::state<state_decode_callback_decision> <= sml::state<state_decode_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_out<decode_run>{} ]
          / action::effect_store_error_out<decode_run>{}
      , sml::state<state_decode_callback_decision> <= sml::state<state_decode_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_out<decode_run>{} ]
      , sml::state<state_decode_failed_callback_decision> <= sml::state<state_decode_failed_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_out<decode_run>{} ]
          / action::effect_store_error_out<decode_run>{}
      , sml::state<state_decode_failed_callback_decision> <= sml::state<state_decode_failed_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_out<decode_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_decode_callback_decision>
          + sml::completion<decode_run> [ guard::guard_has_done_callback<decode_run>{} ]
          / action::effect_emit_decode_done{}
      , sml::state<state_session_ready> <= sml::state<state_decode_callback_decision>
          + sml::completion<decode_run> [ guard::guard_no_done_callback<decode_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_decode_failed_callback_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_callback<decode_run>{} ]
          / action::effect_emit_decode_error{}
      , sml::state<state_session_ready> <= sml::state<state_decode_failed_callback_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_callback<decode_run>{} ]

      //------------------------------------------------------------------------------//
      // Stream rewind.
      , sml::state<state_session_ready> <= sml::state<state_session_ready>
          + sml::event<event::reset_stream_run>
          / action::effect_reset_stream{}

      //------------------------------------------------------------------------------//
      // Requests before initialization answer with explicit errors.
      , sml::state<state_uninit_encode_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<encode_run>
          / action::effect_mark_not_initialized<encode_run>{}
      , sml::state<state_uninit_encode_callback_decision> <= sml::state<state_uninit_encode_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_out<encode_run>{} ]
          / action::effect_store_error_out<encode_run>{}
      , sml::state<state_uninit_encode_callback_decision> <= sml::state<state_uninit_encode_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_out<encode_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_encode_callback_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_callback<encode_run>{} ]
          / action::effect_emit_encode_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_encode_callback_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_callback<encode_run>{} ]
      , sml::state<state_uninit_decode_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<decode_run>
          / action::effect_mark_not_initialized<decode_run>{}
      , sml::state<state_uninit_decode_callback_decision> <= sml::state<state_uninit_decode_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_out<decode_run>{} ]
          / action::effect_store_error_out<decode_run>{}
      , sml::state<state_uninit_decode_callback_decision> <= sml::state<state_uninit_decode_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_out<decode_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_decode_callback_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_callback<decode_run>{} ]
          / action::effect_emit_decode_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_decode_callback_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_callback<decode_run>{} ]

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_session_ready>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_session_ready>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_encode_request_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_encode_request_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decode_request_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decode_request_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decode_codes_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decode_codes_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_encoding>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_encoding>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_quantizing>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_quantizing>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_dequantizing>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_dequantizing>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decoding>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decoding>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_encode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_encode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_encode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_encode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_encode_failed_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_encode_failed_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_encode_failed_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_encode_failed_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decode_failed_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decode_failed_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_session_ready> <= sml::state<state_decode_failed_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_session_ready> <= sml::state<state_decode_failed_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_bind_contract_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_bind_contract_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_bind_capacity_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_bind_capacity_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_binding>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_binding>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_init_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_init_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_init_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_init_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_encode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_encode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_encode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_encode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_decode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_decode_error_out_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_decode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_decode_callback_decision>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;

  bool process_event(const event::initialize &ev) {
    event::initialize_ctx ctx{};
    event::initialize_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::encode_frame &ev) {
    event::encode_frame_ctx ctx{};
    event::encode_frame_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::decode_frame &ev) {
    event::decode_frame_ctx ctx{};
    event::decode_frame_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::reset_stream &ev) {
    event::reset_stream_ctx ctx{};
    event::reset_stream_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }
};

} // namespace emel::speech::codec::mimi
