#pragma once
// benchmark: designed

#include <memory>

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/generator/moshi/actions.hpp"
#include "emel/speech/generator/moshi/context.hpp"
#include "emel/speech/generator/moshi/events.hpp"
#include "emel/speech/generator/moshi/guards.hpp"

namespace emel::speech::generator::moshi {

struct state_uninitialized {};
struct state_bind_contract_decision {};
struct state_binding {};
struct state_lmgen_mode_decision {};
struct state_configure_lmgen {};
struct state_reserve_memory {};
struct state_reserve_memory_decision {};
struct state_allocate_sequence {};
struct state_allocate_sequence_decision {};
struct state_init_error_out_decision {};
struct state_init_callback_decision {};
struct state_init_failed_error_out_decision {};
struct state_init_failed_callback_decision {};
struct state_session_ready {};
struct state_voice_request_decision {};
struct state_voice_error_out_decision {};
struct state_voice_callback_decision {};
struct state_voice_failed_error_out_decision {};
struct state_voice_failed_callback_decision {};
struct state_prefill_voice_request_decision {};
struct state_prefill_voice_allocate_slot {};
struct state_prefill_voice_allocate_slot_decision {};
struct state_prefill_voice_capture_memory {};
struct state_prefill_voice_capture_memory_decision {};
struct state_prefill_voice_begin {};
struct state_prefill_voice_embedding_frame {};
struct state_prefill_voice_embedding_frame_decision {};
struct state_prefill_voice_graph_runtime_decision {};
struct state_prefill_voice_running_graph {};
struct state_prefill_voice_graph_error_out_decision {};
struct state_prefill_voice_graph_result_decision {};
struct state_prefill_voice_advance {};
struct state_prefill_voice_complete_decision {};
struct state_prefill_voice_output_decision {};
struct state_prefill_voice_remaining_out_decision {};
struct state_prefill_voice_error_out_decision {};
struct state_prefill_voice_callback_decision {};
struct state_prefill_voice_failed_error_out_decision {};
struct state_prefill_voice_failed_callback_decision {};
struct state_personaplex_prompt_begin_decision {};
struct state_personaplex_prompt_begin_error_out_decision {};
struct state_personaplex_prompt_begin_callback_decision {};
struct state_personaplex_prompt_begin_failed_error_out_decision {};
struct state_personaplex_prompt_begin_failed_callback_decision {};
struct state_prefill_personaplex_prompt_request_decision {};
struct state_prefill_personaplex_prompt_allocate_slot {};
struct state_prefill_personaplex_prompt_allocate_slot_decision {};
struct state_prefill_personaplex_prompt_capture_memory {};
struct state_prefill_personaplex_prompt_capture_memory_decision {};
struct state_prefill_personaplex_prompt_begin {};
struct state_prefill_personaplex_prompt_phase_decision {};
struct state_prefill_personaplex_prompt_frame {};
struct state_prefill_personaplex_prompt_write_input {};
struct state_prefill_personaplex_prompt_graph_runtime_decision {};
struct state_prefill_personaplex_prompt_running_graph {};
struct state_prefill_personaplex_prompt_graph_error_out_decision {};
struct state_prefill_personaplex_prompt_graph_result_decision {};
struct state_prefill_personaplex_prompt_advance {};
struct state_prefill_personaplex_prompt_complete_decision {};
struct state_prefill_personaplex_prompt_output_decision {};
struct state_prefill_personaplex_prompt_remaining_out_decision {};
struct state_prefill_personaplex_prompt_error_out_decision {};
struct state_prefill_personaplex_prompt_callback_decision {};
struct state_prefill_personaplex_prompt_failed_error_out_decision {};
struct state_prefill_personaplex_prompt_failed_callback_decision {};
struct state_step_request_decision {};
struct state_allocate_step_slot {};
struct state_allocate_step_slot_decision {};
struct state_capture_memory {};
struct state_capture_memory_decision {};
struct state_step_begin {};
struct state_input_write_decision {};
struct state_build_input_sequence {};
struct state_graph_runtime_decision {};
struct state_running_graph {};
struct state_graph_error_out_decision {};
struct state_graph_result_decision {};
struct state_depformer_decision {};
struct state_apply_delay_mask {};
struct state_write_generated_decision {};
struct state_skip_delay_decision {};
struct state_collect_output {};
struct state_output_decision {};
struct state_step_produced_out_decision {};
struct state_step_error_out_decision {};
struct state_step_callback_decision {};
struct state_step_failed_error_out_decision {};
struct state_step_failed_callback_decision {};
struct state_uninit_step_error_out_decision {};
struct state_uninit_step_callback_decision {};
struct state_uninit_voice_error_out_decision {};
struct state_uninit_voice_callback_decision {};
struct state_uninit_prefill_voice_error_out_decision {};
struct state_uninit_prefill_voice_callback_decision {};
struct state_uninit_begin_personaplex_prompt_error_out_decision {};
struct state_uninit_begin_personaplex_prompt_callback_decision {};
struct state_uninit_prefill_personaplex_prompt_error_out_decision {};
struct state_uninit_prefill_personaplex_prompt_callback_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using init_run = event::initialize_run;
    using begin_prompt_run = event::begin_personaplex_prompt_run;
    using load_voice_run = event::load_voice_run;
    using prefill_prompt_run = event::prefill_personaplex_prompt_run;
    using prefill_voice_run = event::prefill_voice_run;
    using step_run = event::step_run;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialization: Moshi contract, then Cortext-owned memory reservation.
        sml::state<state_bind_contract_decision> <= *sml::state<state_uninitialized>
          + sml::event<init_run>
      , sml::state<state_binding> <= sml::state<state_bind_contract_decision>
          + sml::completion<init_run> [ guard::guard_bind_contract_valid{} ]
          / action::effect_bind_contract{}
      , sml::state<state_init_failed_error_out_decision> <= sml::state<state_bind_contract_decision>
          + sml::completion<init_run> [ guard::guard_bind_contract_invalid{} ]
          / action::effect_mark_bind_failed{}
      , sml::state<state_lmgen_mode_decision> <= sml::state<state_binding>
          + sml::completion<init_run>
      , sml::state<state_configure_lmgen> <= sml::state<state_lmgen_mode_decision>
          + sml::completion<init_run> [ guard::guard_personaplex_lmgen{} ]
          / action::effect_configure_personaplex_lmgen{}
      , sml::state<state_configure_lmgen> <= sml::state<state_lmgen_mode_decision>
          + sml::completion<init_run> [ guard::guard_standard_lmgen{} ]
          / action::effect_configure_standard_lmgen{}
      , sml::state<state_reserve_memory> <= sml::state<state_configure_lmgen>
          + sml::completion<init_run>
          / action::effect_reserve_memory{}
      , sml::state<state_reserve_memory_decision> <= sml::state<state_reserve_memory>
          + sml::completion<init_run>
      , sml::state<state_allocate_sequence> <= sml::state<state_reserve_memory_decision>
          + sml::completion<init_run> [ guard::guard_memory_accepted{} ]
          / action::effect_allocate_sequence{}
      , sml::state<state_init_failed_error_out_decision> <= sml::state<state_reserve_memory_decision>
          + sml::completion<init_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<init_run>{}
      , sml::state<state_allocate_sequence_decision> <= sml::state<state_allocate_sequence>
          + sml::completion<init_run>
      , sml::state<state_init_error_out_decision> <= sml::state<state_allocate_sequence_decision>
          + sml::completion<init_run> [ guard::guard_memory_accepted{} ]
      , sml::state<state_init_failed_error_out_decision> <= sml::state<state_allocate_sequence_decision>
          + sml::completion<init_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<init_run>{}

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
      // PersonaPlex voice prompt: bind NATF0/voice GGUF first, then consume
      // one prompt embedding frame per top-level RTC prefill dispatch.
      , sml::state<state_voice_request_decision> <= sml::state<state_session_ready>
          + sml::event<load_voice_run>
      , sml::state<state_voice_error_out_decision> <= sml::state<state_voice_request_decision>
          + sml::completion<load_voice_run> [ guard::guard_voice_contract_valid{} ]
          / action::effect_bind_voice_contract{}
      , sml::state<state_voice_failed_error_out_decision> <= sml::state<state_voice_request_decision>
          + sml::completion<load_voice_run> [ guard::guard_voice_contract_invalid{} ]
          / action::effect_mark_voice_contract_error<load_voice_run>{}
      , sml::state<state_voice_callback_decision> <= sml::state<state_voice_error_out_decision>
          + sml::completion<load_voice_run> [ guard::guard_has_error_out<load_voice_run>{} ]
          / action::effect_store_error_out<load_voice_run>{}
      , sml::state<state_voice_callback_decision> <= sml::state<state_voice_error_out_decision>
          + sml::completion<load_voice_run> [ guard::guard_no_error_out<load_voice_run>{} ]
      , sml::state<state_voice_failed_callback_decision> <= sml::state<state_voice_failed_error_out_decision>
          + sml::completion<load_voice_run> [ guard::guard_has_error_out<load_voice_run>{} ]
          / action::effect_store_error_out<load_voice_run>{}
      , sml::state<state_voice_failed_callback_decision> <= sml::state<state_voice_failed_error_out_decision>
          + sml::completion<load_voice_run> [ guard::guard_no_error_out<load_voice_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_voice_callback_decision>
          + sml::completion<load_voice_run> [ guard::guard_has_done_callback<load_voice_run>{} ]
          / action::effect_emit_load_voice_done{}
      , sml::state<state_session_ready> <= sml::state<state_voice_callback_decision>
          + sml::completion<load_voice_run> [ guard::guard_no_done_callback<load_voice_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_voice_failed_callback_decision>
          + sml::completion<load_voice_run> [ guard::guard_has_error_callback<load_voice_run>{} ]
          / action::effect_emit_load_voice_error{}
      , sml::state<state_session_ready> <= sml::state<state_voice_failed_callback_decision>
          + sml::completion<load_voice_run> [ guard::guard_no_error_callback<load_voice_run>{} ]

      , sml::state<state_prefill_voice_request_decision> <= sml::state<state_session_ready>
          + sml::event<prefill_voice_run>
      , sml::state<state_prefill_voice_allocate_slot> <= sml::state<state_prefill_voice_request_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_prefill_request_valid{} ]
          / action::effect_allocate_voice_prefill_slot{}
      , sml::state<state_prefill_voice_failed_error_out_decision> <= sml::state<state_prefill_voice_request_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_prefill_request_invalid{} ]
          / action::effect_mark_voice_contract_error<prefill_voice_run>{}
      , sml::state<state_prefill_voice_allocate_slot_decision> <= sml::state<state_prefill_voice_allocate_slot>
          + sml::completion<prefill_voice_run>
      , sml::state<state_prefill_voice_capture_memory> <= sml::state<state_prefill_voice_allocate_slot_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_memory_accepted{} ]
          / action::effect_capture_voice_prefill_memory{}
      , sml::state<state_prefill_voice_failed_error_out_decision> <= sml::state<state_prefill_voice_allocate_slot_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<prefill_voice_run>{}
      , sml::state<state_prefill_voice_capture_memory_decision> <= sml::state<state_prefill_voice_capture_memory>
          + sml::completion<prefill_voice_run>
      , sml::state<state_prefill_voice_begin> <= sml::state<state_prefill_voice_capture_memory_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_memory_accepted{} ]
          / action::effect_begin_voice_prefill{}
      , sml::state<state_prefill_voice_failed_error_out_decision> <= sml::state<state_prefill_voice_capture_memory_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<prefill_voice_run>{}
      , sml::state<state_prefill_voice_embedding_frame> <= sml::state<state_prefill_voice_begin>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_embedding_frame_f32{} ]
          / action::effect_load_voice_embedding_frame_f32{}
      , sml::state<state_prefill_voice_embedding_frame> <= sml::state<state_prefill_voice_begin>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_embedding_frame_f16{} ]
          / action::effect_load_voice_embedding_frame_f16{}
      , sml::state<state_prefill_voice_embedding_frame> <= sml::state<state_prefill_voice_begin>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_embedding_frame_bf16{} ]
          / action::effect_load_voice_embedding_frame_bf16{}
      , sml::state<state_prefill_voice_embedding_frame_decision> <= sml::state<state_prefill_voice_embedding_frame>
          + sml::completion<prefill_voice_run>
      , sml::state<state_prefill_voice_graph_runtime_decision> <= sml::state<state_prefill_voice_embedding_frame_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_embedding_frame_loaded{} ]
      , sml::state<state_prefill_voice_failed_error_out_decision> <= sml::state<state_prefill_voice_embedding_frame_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_embedding_frame_failed{} ]
          / action::effect_mark_voice_contract_error<prefill_voice_run>{}
      , sml::state<state_prefill_voice_running_graph> <= sml::state<state_prefill_voice_graph_runtime_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_graph_runtime_available{} ]
          / action::effect_run_voice_graph_runtime{}
      , sml::state<state_prefill_voice_failed_error_out_decision> <= sml::state<state_prefill_voice_graph_runtime_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_graph_runtime_unavailable{} ]
          / action::effect_mark_graph_runtime_unavailable<prefill_voice_run>{}
      , sml::state<state_prefill_voice_graph_error_out_decision> <= sml::state<state_prefill_voice_running_graph>
          + sml::completion<prefill_voice_run>
      , sml::state<state_prefill_voice_graph_result_decision> <= sml::state<state_prefill_voice_graph_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_graph_error_out{} ]
          / action::effect_store_graph_error_out{}
      , sml::state<state_prefill_voice_graph_result_decision> <= sml::state<state_prefill_voice_graph_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_graph_error_out{} ]
      , sml::state<state_prefill_voice_advance> <= sml::state<state_prefill_voice_graph_result_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_graph_step_accepted{} ]
          / action::effect_advance_voice_prefill{}
      , sml::state<state_prefill_voice_failed_error_out_decision> <= sml::state<state_prefill_voice_graph_result_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_graph_step_rejected{} ]
          / action::effect_mark_graph_runtime_error<prefill_voice_run>{}
      , sml::state<state_prefill_voice_complete_decision> <= sml::state<state_prefill_voice_advance>
          + sml::completion<prefill_voice_run>
      , sml::state<state_prefill_voice_output_decision> <= sml::state<state_prefill_voice_complete_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_prefill_complete{} ]
          / action::effect_copy_voice_cache{}
      , sml::state<state_prefill_voice_output_decision> <= sml::state<state_prefill_voice_complete_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_voice_prefill_pending{} ]
      , sml::state<state_prefill_voice_remaining_out_decision> <= sml::state<state_prefill_voice_output_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_voice_complete_out{} ]
          / action::effect_store_voice_complete_out{}
      , sml::state<state_prefill_voice_remaining_out_decision> <= sml::state<state_prefill_voice_output_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_voice_complete_out{} ]
      , sml::state<state_prefill_voice_error_out_decision> <= sml::state<state_prefill_voice_remaining_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_voice_remaining_out{} ]
          / action::effect_store_voice_remaining_out{}
      , sml::state<state_prefill_voice_error_out_decision> <= sml::state<state_prefill_voice_remaining_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_voice_remaining_out{} ]
      , sml::state<state_prefill_voice_callback_decision> <= sml::state<state_prefill_voice_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_error_out<prefill_voice_run>{} ]
          / action::effect_store_error_out<prefill_voice_run>{}
      , sml::state<state_prefill_voice_callback_decision> <= sml::state<state_prefill_voice_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_error_out<prefill_voice_run>{} ]
      , sml::state<state_prefill_voice_failed_callback_decision> <= sml::state<state_prefill_voice_failed_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_error_out<prefill_voice_run>{} ]
          / action::effect_store_error_out<prefill_voice_run>{}
      , sml::state<state_prefill_voice_failed_callback_decision> <= sml::state<state_prefill_voice_failed_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_error_out<prefill_voice_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_prefill_voice_callback_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_done_callback<prefill_voice_run>{} ]
          / action::effect_emit_prefill_voice_done{}
      , sml::state<state_session_ready> <= sml::state<state_prefill_voice_callback_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_done_callback<prefill_voice_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_prefill_voice_failed_callback_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_error_callback<prefill_voice_run>{} ]
          / action::effect_emit_prefill_voice_error{}
      , sml::state<state_session_ready> <= sml::state<state_prefill_voice_failed_callback_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_error_callback<prefill_voice_run>{} ]

      //------------------------------------------------------------------------------//
      // PersonaPlex system prompt: after voice embeddings/cache are loaded, consume
      // the reference silence/text/silence prompt frames one RTC dispatch at a time.
      , sml::state<state_personaplex_prompt_begin_decision> <= sml::state<state_session_ready>
          + sml::event<begin_prompt_run>
      , sml::state<state_personaplex_prompt_begin_error_out_decision> <= sml::state<state_personaplex_prompt_begin_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_personaplex_prompt_begin_nonempty_valid{} ]
          / action::effect_bind_personaplex_prompt{}
      , sml::state<state_personaplex_prompt_begin_error_out_decision> <= sml::state<state_personaplex_prompt_begin_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_personaplex_prompt_begin_empty_valid{} ]
          / action::effect_bind_empty_personaplex_prompt{}
      , sml::state<state_personaplex_prompt_begin_failed_error_out_decision> <= sml::state<state_personaplex_prompt_begin_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_personaplex_prompt_begin_invalid{} ]
          / action::effect_mark_personaplex_prompt_error<begin_prompt_run>{}
      , sml::state<state_personaplex_prompt_begin_callback_decision> <= sml::state<state_personaplex_prompt_begin_error_out_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_has_error_out<begin_prompt_run>{} ]
          / action::effect_store_error_out<begin_prompt_run>{}
      , sml::state<state_personaplex_prompt_begin_callback_decision> <= sml::state<state_personaplex_prompt_begin_error_out_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_no_error_out<begin_prompt_run>{} ]
      , sml::state<state_personaplex_prompt_begin_failed_callback_decision> <= sml::state<state_personaplex_prompt_begin_failed_error_out_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_has_error_out<begin_prompt_run>{} ]
          / action::effect_store_error_out<begin_prompt_run>{}
      , sml::state<state_personaplex_prompt_begin_failed_callback_decision> <= sml::state<state_personaplex_prompt_begin_failed_error_out_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_no_error_out<begin_prompt_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_personaplex_prompt_begin_callback_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_has_done_callback<begin_prompt_run>{} ]
          / action::effect_emit_begin_personaplex_prompt_done{}
      , sml::state<state_session_ready> <= sml::state<state_personaplex_prompt_begin_callback_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_no_done_callback<begin_prompt_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_personaplex_prompt_begin_failed_callback_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_has_error_callback<begin_prompt_run>{} ]
          / action::effect_emit_begin_personaplex_prompt_error{}
      , sml::state<state_session_ready> <= sml::state<state_personaplex_prompt_begin_failed_callback_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_no_error_callback<begin_prompt_run>{} ]

      , sml::state<state_prefill_personaplex_prompt_request_decision> <= sml::state<state_session_ready>
          + sml::event<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_allocate_slot> <= sml::state<state_prefill_personaplex_prompt_request_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_prefill_request_valid{} ]
          / action::effect_allocate_personaplex_prompt_slot{}
      , sml::state<state_prefill_personaplex_prompt_failed_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_request_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_prefill_request_invalid{} ]
          / action::effect_mark_personaplex_prompt_error<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_allocate_slot_decision> <= sml::state<state_prefill_personaplex_prompt_allocate_slot>
          + sml::completion<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_capture_memory> <= sml::state<state_prefill_personaplex_prompt_allocate_slot_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_memory_accepted{} ]
          / action::effect_capture_personaplex_prompt_memory{}
      , sml::state<state_prefill_personaplex_prompt_failed_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_allocate_slot_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_capture_memory_decision> <= sml::state<state_prefill_personaplex_prompt_capture_memory>
          + sml::completion<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_begin> <= sml::state<state_prefill_personaplex_prompt_capture_memory_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_memory_accepted{} ]
          / action::effect_begin_personaplex_prompt_prefill{}
      , sml::state<state_prefill_personaplex_prompt_failed_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_capture_memory_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_phase_decision> <= sml::state<state_prefill_personaplex_prompt_begin>
          + sml::completion<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_frame> <= sml::state<state_prefill_personaplex_prompt_phase_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_pre_silence_pending{} ]
          / action::effect_build_personaplex_prompt_silence_frame{}
      , sml::state<state_prefill_personaplex_prompt_frame> <= sml::state<state_prefill_personaplex_prompt_phase_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_text_pending{} ]
          / action::effect_build_personaplex_prompt_text_frame{}
      , sml::state<state_prefill_personaplex_prompt_frame> <= sml::state<state_prefill_personaplex_prompt_phase_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_post_silence_pending{} ]
          / action::effect_build_personaplex_prompt_silence_frame{}
      , sml::state<state_prefill_personaplex_prompt_failed_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_phase_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_phase_invalid{} ]
          / action::effect_mark_personaplex_prompt_error<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_write_input> <= sml::state<state_prefill_personaplex_prompt_frame>
          + sml::completion<prefill_prompt_run>
          / action::effect_write_and_build_personaplex_prompt_input{}
      , sml::state<state_prefill_personaplex_prompt_graph_runtime_decision> <= sml::state<state_prefill_personaplex_prompt_write_input>
          + sml::completion<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_running_graph> <= sml::state<state_prefill_personaplex_prompt_graph_runtime_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_graph_runtime_available{} ]
          / action::effect_run_personaplex_prompt_graph_runtime{}
      , sml::state<state_prefill_personaplex_prompt_failed_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_graph_runtime_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_graph_runtime_unavailable{} ]
          / action::effect_mark_graph_runtime_unavailable<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_graph_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_running_graph>
          + sml::completion<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_graph_result_decision> <= sml::state<state_prefill_personaplex_prompt_graph_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_graph_error_out{} ]
          / action::effect_store_graph_error_out{}
      , sml::state<state_prefill_personaplex_prompt_graph_result_decision> <= sml::state<state_prefill_personaplex_prompt_graph_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_graph_error_out{} ]
      , sml::state<state_prefill_personaplex_prompt_advance> <= sml::state<state_prefill_personaplex_prompt_graph_result_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_graph_step_accepted_personaplex_prompt_pre_silence{} ]
          / action::effect_advance_personaplex_prompt_pre_silence{}
      , sml::state<state_prefill_personaplex_prompt_advance> <= sml::state<state_prefill_personaplex_prompt_graph_result_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_graph_step_accepted_personaplex_prompt_text{} ]
          / action::effect_advance_personaplex_prompt_text{}
      , sml::state<state_prefill_personaplex_prompt_advance> <= sml::state<state_prefill_personaplex_prompt_graph_result_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_graph_step_accepted_personaplex_prompt_post_silence{} ]
          / action::effect_advance_personaplex_prompt_post_silence{}
      , sml::state<state_prefill_personaplex_prompt_failed_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_graph_result_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_graph_step_rejected{} ]
          / action::effect_mark_graph_runtime_error<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_complete_decision> <= sml::state<state_prefill_personaplex_prompt_advance>
          + sml::completion<prefill_prompt_run>
      , sml::state<state_prefill_personaplex_prompt_output_decision> <= sml::state<state_prefill_personaplex_prompt_complete_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_complete{} ]
          / action::effect_finish_personaplex_prompt{}
      , sml::state<state_prefill_personaplex_prompt_output_decision> <= sml::state<state_prefill_personaplex_prompt_complete_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_personaplex_prompt_pending{} ]
          / action::effect_publish_personaplex_prompt_pending{}
      , sml::state<state_prefill_personaplex_prompt_remaining_out_decision> <= sml::state<state_prefill_personaplex_prompt_output_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_personaplex_prompt_complete_out{} ]
          / action::effect_store_personaplex_prompt_complete_out{}
      , sml::state<state_prefill_personaplex_prompt_remaining_out_decision> <= sml::state<state_prefill_personaplex_prompt_output_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_personaplex_prompt_complete_out{} ]
      , sml::state<state_prefill_personaplex_prompt_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_remaining_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_personaplex_prompt_remaining_out{} ]
          / action::effect_store_personaplex_prompt_remaining_out{}
      , sml::state<state_prefill_personaplex_prompt_error_out_decision> <= sml::state<state_prefill_personaplex_prompt_remaining_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_personaplex_prompt_remaining_out{} ]
      , sml::state<state_prefill_personaplex_prompt_callback_decision> <= sml::state<state_prefill_personaplex_prompt_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_error_out<prefill_prompt_run>{} ]
          / action::effect_store_error_out<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_callback_decision> <= sml::state<state_prefill_personaplex_prompt_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_error_out<prefill_prompt_run>{} ]
      , sml::state<state_prefill_personaplex_prompt_failed_callback_decision> <= sml::state<state_prefill_personaplex_prompt_failed_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_error_out<prefill_prompt_run>{} ]
          / action::effect_store_error_out<prefill_prompt_run>{}
      , sml::state<state_prefill_personaplex_prompt_failed_callback_decision> <= sml::state<state_prefill_personaplex_prompt_failed_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_error_out<prefill_prompt_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_prefill_personaplex_prompt_callback_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_done_callback<prefill_prompt_run>{} ]
          / action::effect_emit_prefill_personaplex_prompt_done{}
      , sml::state<state_session_ready> <= sml::state<state_prefill_personaplex_prompt_callback_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_done_callback<prefill_prompt_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_prefill_personaplex_prompt_failed_callback_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_error_callback<prefill_prompt_run>{} ]
          / action::effect_emit_prefill_personaplex_prompt_error{}
      , sml::state<state_session_ready> <= sml::state<state_prefill_personaplex_prompt_failed_callback_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_error_callback<prefill_prompt_run>{} ]

      //------------------------------------------------------------------------------//
      // Online step: validate audio-token frame, allocate the next
      // Cortext-owned KV slot, capture memory, then enter the explicitly
      // modeled Moshi graph execution route.
      , sml::state<state_step_request_decision> <= sml::state<state_session_ready>
          + sml::event<step_run>
      , sml::state<state_allocate_step_slot> <= sml::state<state_step_request_decision>
          + sml::completion<step_run> [ guard::guard_step_request_valid{} ]
          / action::effect_allocate_step_slot{}
      , sml::state<state_step_failed_error_out_decision> <= sml::state<state_step_request_decision>
          + sml::completion<step_run> [ guard::guard_step_blocked_by_voice_prompt{} ]
          / action::effect_mark_voice_prompt_pending{}
      , sml::state<state_step_failed_error_out_decision> <= sml::state<state_step_request_decision>
          + sml::completion<step_run> [ guard::guard_step_request_shape_invalid{} ]
          / action::effect_mark_step_request_invalid{}
      , sml::state<state_allocate_step_slot_decision> <= sml::state<state_allocate_step_slot>
          + sml::completion<step_run>
      , sml::state<state_capture_memory> <= sml::state<state_allocate_step_slot_decision>
          + sml::completion<step_run> [ guard::guard_memory_accepted{} ]
          / action::effect_capture_memory{}
      , sml::state<state_step_failed_error_out_decision> <= sml::state<state_allocate_step_slot_decision>
          + sml::completion<step_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<step_run>{}
      , sml::state<state_capture_memory_decision> <= sml::state<state_capture_memory>
          + sml::completion<step_run>
      , sml::state<state_step_begin> <= sml::state<state_capture_memory_decision>
          + sml::completion<step_run> [ guard::guard_memory_accepted{} ]
          / action::effect_begin_step{}
      , sml::state<state_step_failed_error_out_decision> <= sml::state<state_capture_memory_decision>
          + sml::completion<step_run> [ guard::guard_memory_rejected{} ]
          / action::effect_mark_memory_error<step_run>{}
      , sml::state<state_input_write_decision> <= sml::state<state_step_begin>
          + sml::completion<step_run>
      , sml::state<state_build_input_sequence> <= sml::state<state_input_write_decision>
          + sml::completion<step_run> [ guard::guard_has_full_input{} ]
          / action::effect_write_full_input{}
      , sml::state<state_build_input_sequence> <= sml::state<state_input_write_decision>
          + sml::completion<step_run> [ guard::guard_has_tail_input{} ]
          / action::effect_write_tail_input{}
      , sml::state<state_build_input_sequence> <= sml::state<state_input_write_decision>
          + sml::completion<step_run> [ guard::guard_has_no_input_write{} ]
      , sml::state<state_graph_runtime_decision> <= sml::state<state_build_input_sequence>
          + sml::completion<step_run>
          / action::effect_build_input_sequence{}
      , sml::state<state_running_graph> <= sml::state<state_graph_runtime_decision>
          + sml::completion<step_run> [ guard::guard_graph_runtime_available{} ]
          / action::effect_run_graph_runtime{}
      , sml::state<state_step_failed_error_out_decision> <= sml::state<state_graph_runtime_decision>
          + sml::completion<step_run> [ guard::guard_graph_runtime_unavailable{} ]
          / action::effect_mark_graph_runtime_unavailable<step_run>{}
      , sml::state<state_graph_error_out_decision> <= sml::state<state_running_graph>
          + sml::completion<step_run>
      , sml::state<state_graph_result_decision> <= sml::state<state_graph_error_out_decision>
          + sml::completion<step_run> [ guard::guard_has_graph_error_out{} ]
          / action::effect_store_graph_error_out{}
      , sml::state<state_graph_result_decision> <= sml::state<state_graph_error_out_decision>
          + sml::completion<step_run> [ guard::guard_no_graph_error_out{} ]
      , sml::state<state_depformer_decision> <= sml::state<state_graph_result_decision>
          + sml::completion<step_run> [ guard::guard_graph_step_accepted{} ]
      , sml::state<state_step_failed_error_out_decision> <= sml::state<state_graph_result_decision>
          + sml::completion<step_run> [ guard::guard_graph_step_rejected{} ]
          / action::effect_mark_graph_runtime_error<step_run>{}
      , sml::state<state_apply_delay_mask> <= sml::state<state_depformer_decision>
          + sml::completion<step_run> [ guard::guard_replace_depformer_tokens{} ]
          / action::effect_replace_audio_with_zero{}
      , sml::state<state_apply_delay_mask> <= sml::state<state_depformer_decision>
          + sml::completion<step_run> [ guard::guard_use_graph_audio_tokens{} ]
      , sml::state<state_write_generated_decision> <= sml::state<state_apply_delay_mask>
          + sml::completion<step_run>
          / action::effect_apply_delay_mask_and_advance{}
      , sml::state<state_skip_delay_decision> <= sml::state<state_write_generated_decision>
          + sml::completion<step_run> [ guard::guard_should_write_generated{} ]
          / action::effect_write_generated{}
      , sml::state<state_skip_delay_decision> <= sml::state<state_write_generated_decision>
          + sml::completion<step_run> [ guard::guard_provided_input{} ]
          / action::effect_write_generated_missing{}
      , sml::state<state_step_produced_out_decision> <= sml::state<state_skip_delay_decision>
          + sml::completion<step_run> [ guard::guard_skip_pending{} ]
          / action::effect_consume_skip{}
      , sml::state<state_step_produced_out_decision> <= sml::state<state_skip_delay_decision>
          + sml::completion<step_run> [ guard::guard_before_delay{} ]
          / action::effect_publish_no_output{}
      , sml::state<state_collect_output> <= sml::state<state_skip_delay_decision>
          + sml::completion<step_run> [ guard::guard_past_delay{} ]
          / action::effect_collect_delayed_output{}
      , sml::state<state_output_decision> <= sml::state<state_collect_output>
          + sml::completion<step_run>
      , sml::state<state_step_produced_out_decision> <= sml::state<state_output_decision>
          + sml::completion<step_run> [ guard::guard_output_has_ungenerated{} ]
          / action::effect_publish_no_output{}
      , sml::state<state_step_produced_out_decision> <= sml::state<state_output_decision>
          + sml::completion<step_run> [ guard::guard_output_complete{} ]
          / action::effect_publish_output{}

      , sml::state<state_step_error_out_decision> <= sml::state<state_step_produced_out_decision>
          + sml::completion<step_run> [ guard::guard_has_produced_out{} ]
          / action::effect_store_produced_out{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_step_produced_out_decision>
          + sml::completion<step_run> [ guard::guard_no_produced_out{} ]

      , sml::state<state_step_callback_decision> <= sml::state<state_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_has_error_out<step_run>{} ]
          / action::effect_store_error_out<step_run>{}
      , sml::state<state_step_callback_decision> <= sml::state<state_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_no_error_out<step_run>{} ]
      , sml::state<state_step_failed_callback_decision> <= sml::state<state_step_failed_error_out_decision>
          + sml::completion<step_run> [ guard::guard_has_error_out<step_run>{} ]
          / action::effect_store_error_out<step_run>{}
      , sml::state<state_step_failed_callback_decision> <= sml::state<state_step_failed_error_out_decision>
          + sml::completion<step_run> [ guard::guard_no_error_out<step_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_step_callback_decision>
          + sml::completion<step_run> [ guard::guard_has_done_callback<step_run>{} ]
          / action::effect_emit_step_done{}
      , sml::state<state_session_ready> <= sml::state<state_step_callback_decision>
          + sml::completion<step_run> [ guard::guard_no_done_callback<step_run>{} ]
      , sml::state<state_session_ready> <= sml::state<state_step_failed_callback_decision>
          + sml::completion<step_run> [ guard::guard_has_error_callback<step_run>{} ]
          / action::effect_emit_step_error{}
      , sml::state<state_session_ready> <= sml::state<state_step_failed_callback_decision>
          + sml::completion<step_run> [ guard::guard_no_error_callback<step_run>{} ]

      //------------------------------------------------------------------------------//
      // Requests before initialization answer with explicit errors.
      , sml::state<state_uninit_step_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<step_run>
          / action::effect_mark_not_initialized<step_run>{}
      , sml::state<state_uninit_step_callback_decision> <= sml::state<state_uninit_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_has_error_out<step_run>{} ]
          / action::effect_store_error_out<step_run>{}
      , sml::state<state_uninit_step_callback_decision> <= sml::state<state_uninit_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_no_error_out<step_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_step_callback_decision>
          + sml::completion<step_run> [ guard::guard_has_error_callback<step_run>{} ]
          / action::effect_emit_step_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_step_callback_decision>
          + sml::completion<step_run> [ guard::guard_no_error_callback<step_run>{} ]

      , sml::state<state_uninit_voice_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<load_voice_run>
          / action::effect_mark_not_initialized<load_voice_run>{}
      , sml::state<state_uninit_voice_callback_decision> <= sml::state<state_uninit_voice_error_out_decision>
          + sml::completion<load_voice_run> [ guard::guard_has_error_out<load_voice_run>{} ]
          / action::effect_store_error_out<load_voice_run>{}
      , sml::state<state_uninit_voice_callback_decision> <= sml::state<state_uninit_voice_error_out_decision>
          + sml::completion<load_voice_run> [ guard::guard_no_error_out<load_voice_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_voice_callback_decision>
          + sml::completion<load_voice_run> [ guard::guard_has_error_callback<load_voice_run>{} ]
          / action::effect_emit_load_voice_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_voice_callback_decision>
          + sml::completion<load_voice_run> [ guard::guard_no_error_callback<load_voice_run>{} ]

      , sml::state<state_uninit_prefill_voice_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<prefill_voice_run>
          / action::effect_mark_not_initialized<prefill_voice_run>{}
      , sml::state<state_uninit_prefill_voice_callback_decision> <= sml::state<state_uninit_prefill_voice_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_error_out<prefill_voice_run>{} ]
          / action::effect_store_error_out<prefill_voice_run>{}
      , sml::state<state_uninit_prefill_voice_callback_decision> <= sml::state<state_uninit_prefill_voice_error_out_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_error_out<prefill_voice_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_prefill_voice_callback_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_has_error_callback<prefill_voice_run>{} ]
          / action::effect_emit_prefill_voice_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_prefill_voice_callback_decision>
          + sml::completion<prefill_voice_run> [ guard::guard_no_error_callback<prefill_voice_run>{} ]

      , sml::state<state_uninit_begin_personaplex_prompt_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<begin_prompt_run>
          / action::effect_mark_not_initialized<begin_prompt_run>{}
      , sml::state<state_uninit_begin_personaplex_prompt_callback_decision> <= sml::state<state_uninit_begin_personaplex_prompt_error_out_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_has_error_out<begin_prompt_run>{} ]
          / action::effect_store_error_out<begin_prompt_run>{}
      , sml::state<state_uninit_begin_personaplex_prompt_callback_decision> <= sml::state<state_uninit_begin_personaplex_prompt_error_out_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_no_error_out<begin_prompt_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_begin_personaplex_prompt_callback_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_has_error_callback<begin_prompt_run>{} ]
          / action::effect_emit_begin_personaplex_prompt_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_begin_personaplex_prompt_callback_decision>
          + sml::completion<begin_prompt_run> [ guard::guard_no_error_callback<begin_prompt_run>{} ]

      , sml::state<state_uninit_prefill_personaplex_prompt_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<prefill_prompt_run>
          / action::effect_mark_not_initialized<prefill_prompt_run>{}
      , sml::state<state_uninit_prefill_personaplex_prompt_callback_decision> <= sml::state<state_uninit_prefill_personaplex_prompt_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_error_out<prefill_prompt_run>{} ]
          / action::effect_store_error_out<prefill_prompt_run>{}
      , sml::state<state_uninit_prefill_personaplex_prompt_callback_decision> <= sml::state<state_uninit_prefill_personaplex_prompt_error_out_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_error_out<prefill_prompt_run>{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninit_prefill_personaplex_prompt_callback_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_has_error_callback<prefill_prompt_run>{} ]
          / action::effect_emit_prefill_personaplex_prompt_error{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_prefill_personaplex_prompt_callback_decision>
          + sml::completion<prefill_prompt_run> [ guard::guard_no_error_callback<prefill_prompt_run>{} ]

      //------------------------------------------------------------------------------//
      // Reset.
      , sml::state<state_uninitialized> <= sml::state<state_session_ready>
          + sml::event<event::reset_run>
          / action::effect_reset_session{}

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
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() : memory_snapshot_(std::make_unique<emel::memory::view::snapshot>()) {}
  explicit sm(const emel::memory::hybrid::kv_binding &kv_cache)
      : base_type(std::in_place, kv_cache),
        memory_snapshot_(std::make_unique<emel::memory::view::snapshot>()) {}
  sm(const emel::memory::hybrid::kv_binding &kv_cache,
     const action::graph_binding &graph_executor)
      : base_type(std::in_place, kv_cache, graph_executor),
        memory_snapshot_(std::make_unique<emel::memory::view::snapshot>()) {}

  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;
  sm(sm &&) = delete;
  sm &operator=(sm &&) = delete;

  bool process_event(const event::initialize &ev) {
    event::initialize_ctx ctx{};
    event::initialize_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::load_voice &ev) {
    event::load_voice_ctx ctx{};
    event::load_voice_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::begin_personaplex_prompt &ev) {
    event::begin_personaplex_prompt_ctx ctx{};
    event::begin_personaplex_prompt_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::prefill_voice &ev) {
    event::prefill_voice_ctx ctx{*memory_snapshot_};
    event::prefill_voice_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::prefill_personaplex_prompt &ev) {
    event::prefill_personaplex_prompt_ctx ctx{*memory_snapshot_};
    event::prefill_personaplex_prompt_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::step &ev) {
    event::step_ctx ctx{*memory_snapshot_};
    event::step_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::reset &ev) {
    event::reset_run runtime_ev{ev};
    return base_type::process_event(runtime_ev);
  }

private:
  std::unique_ptr<emel::memory::view::snapshot> memory_snapshot_ = {};
};

using Generator = sm;

} // namespace emel::speech::generator::moshi
