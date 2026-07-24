#pragma once
// benchmark: designed

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/predictor/moshi/executor/actions.hpp"
#include "emel/speech/predictor/moshi/executor/context.hpp"
#include "emel/speech/predictor/moshi/executor/events.hpp"
#include "emel/speech/predictor/moshi/executor/guards.hpp"

namespace emel::speech::predictor::moshi::executor {

struct state_uninitialized {};
struct state_reset_temporal_positions_decision {};
struct state_reset_temporal_positions_result_decision {};
struct state_reset_depformer_positions_decision {};
struct state_reset_depformer_positions_result_decision {};
struct state_reset_failed {};
struct state_bind_contract_decision {};
struct state_input_embedding_contract_decision {};
struct state_temporal_projection_layout_decision {};
struct state_depformer_projection_layout_decision {};
struct state_sampling_seed_decision {};
struct state_text_sampling_top_k_decision {};
struct state_audio_sampling_top_k_decision {};
struct state_init_error_out_decision {};
struct state_init_callback_decision {};
struct state_init_failed_error_out_decision {};
struct state_init_failed_callback_decision {};
struct state_ready {};
struct state_step_model_decision {};
struct state_step_shape_decision {};
struct state_step_execution_decision {};
struct state_input_embedding {};
struct state_input_text_embedding_bind {};
struct state_input_text_embedding_bind_result_decision {};
struct state_input_text_embedding_row {};
struct state_input_text_embedding_row_result_decision {};
struct state_input_text_embedding_apply {};
struct state_input_text_embedding_result_decision {};
struct state_input_audio_token_decision {};
struct state_input_audio_embedding {};
struct state_input_audio_embedding_bind_result_decision {};
struct state_input_audio_embedding_row {};
struct state_input_audio_embedding_row_result_decision {};
struct state_input_audio_embedding_apply {};
struct state_input_audio_embedding_result_decision {};
struct state_input_audio_codebook_advance {};
struct state_input_embedding_result_decision {};
struct state_bind_temporal_kv {};
struct state_temporal_kv_result_decision {};
struct state_temporal_position_result_decision {};
struct state_temporal_position_valid_decision {};
struct state_temporal_layer_norm {};
struct state_temporal_layer_norm_rms_result_decision {};
struct state_temporal_layer_norm_scale {};
struct state_temporal_layer_norm_result_decision {};
struct state_temporal_layer_projection {};
struct state_temporal_layer_projection_bind_result_decision {};
struct state_temporal_layer_projection_run {};
struct state_temporal_layer_projection_result_decision {};
struct state_temporal_layer_rope {};
struct state_temporal_layer_rope_result_decision {};
struct state_temporal_layer_key_rope {};
struct state_temporal_layer_key_rope_result_decision {};
struct state_temporal_layer_cache_write {};
struct state_temporal_layer_cache_write_result_decision {};
struct state_temporal_layer_attention {};
struct state_temporal_layer_attention_result_decision {};
struct state_temporal_layer_out_projection {};
struct state_temporal_layer_out_projection_bind_result_decision {};
struct state_temporal_layer_out_projection_run {};
struct state_temporal_layer_out_projection_result_decision {};
struct state_temporal_layer_residual {};
struct state_temporal_layer_residual_result_decision {};
struct state_temporal_layer_norm2 {};
struct state_temporal_layer_norm2_rms_result_decision {};
struct state_temporal_layer_norm2_scale {};
struct state_temporal_layer_norm2_result_decision {};
struct state_temporal_layer_gating_in {};
struct state_temporal_layer_gating_in_bind_result_decision {};
struct state_temporal_layer_gating_in_run {};
struct state_temporal_layer_gating_in_result_decision {};
struct state_temporal_layer_silu_gate {};
struct state_temporal_layer_silu_gate_silu_result_decision {};
struct state_temporal_layer_silu_gate_mul {};
struct state_temporal_layer_silu_gate_result_decision {};
struct state_temporal_layer_gating_out {};
struct state_temporal_layer_gating_out_bind_result_decision {};
struct state_temporal_layer_gating_out_run {};
struct state_temporal_layer_gating_out_result_decision {};
struct state_temporal_layer_ff_residual {};
struct state_temporal_layer_ff_residual_result_decision {};
struct state_temporal_layer_advance {};
struct state_temporal_out_norm {};
struct state_temporal_out_norm_rms_result_decision {};
struct state_temporal_out_norm_scale {};
struct state_temporal_out_norm_result_decision {};
struct state_text_logits_phase_decision {};
struct state_sampling_ready {};
struct state_text_logits {};
struct state_text_logits_bind_result_decision {};
struct state_text_logits_run {};
struct state_text_logits_sample_projection {};
struct state_text_logits_sample_projection_result_decision {};
struct state_text_logits_sample_select {};
struct state_text_logits_result_decision {};
struct state_bind_depformer_kv {};
struct state_depformer_kv_result_decision {};
struct state_depformer_position_reset_result_decision {};
struct state_depformer_position_advance_result_decision {};
struct state_depformer_weight_route_decision {};
struct state_depformer_weight_decision {};
struct state_depformer_input {};
struct state_depformer_input_projection_bind_result_decision {};
struct state_depformer_input_projection {};
struct state_depformer_input_projection_result_decision {};
struct state_depformer_input_embedding_bind {};
struct state_depformer_input_embedding_bind_result_decision {};
struct state_depformer_input_embedding_row {};
struct state_depformer_input_embedding_row_result_decision {};
struct state_depformer_input_embedding_apply {};
struct state_depformer_input_result_decision {};
struct state_depformer_layer_norm {};
struct state_depformer_layer_norm_rms_result_decision {};
struct state_depformer_layer_norm_scale {};
struct state_depformer_layer_norm_result_decision {};
struct state_depformer_layer_projection {};
struct state_depformer_layer_projection_bind_result_decision {};
struct state_depformer_layer_projection_run {};
struct state_depformer_layer_projection_result_decision {};
struct state_depformer_layer_cache_write {};
struct state_depformer_layer_cache_write_result_decision {};
struct state_depformer_layer_attention {};
struct state_depformer_layer_attention_result_decision {};
struct state_depformer_layer_out_projection {};
struct state_depformer_layer_out_projection_bind_result_decision {};
struct state_depformer_layer_out_projection_run {};
struct state_depformer_layer_out_projection_result_decision {};
struct state_depformer_layer_residual {};
struct state_depformer_layer_residual_result_decision {};
struct state_depformer_layer_norm2 {};
struct state_depformer_layer_norm2_rms_result_decision {};
struct state_depformer_layer_norm2_scale {};
struct state_depformer_layer_norm2_result_decision {};
struct state_depformer_layer_gating_in {};
struct state_depformer_layer_gating_in_bind_result_decision {};
struct state_depformer_layer_gating_in_run {};
struct state_depformer_layer_gating_in_result_decision {};
struct state_depformer_layer_silu_gate {};
struct state_depformer_layer_silu_gate_silu_result_decision {};
struct state_depformer_layer_silu_gate_mul {};
struct state_depformer_layer_silu_gate_result_decision {};
struct state_depformer_layer_gating_out {};
struct state_depformer_layer_gating_out_bind_result_decision {};
struct state_depformer_layer_gating_out_run {};
struct state_depformer_layer_gating_out_result_decision {};
struct state_depformer_layer_ff_residual {};
struct state_depformer_layer_ff_residual_result_decision {};
struct state_depformer_layer_advance {};
struct state_depformer_logits {};
struct state_depformer_logits_bind_result_decision {};
struct state_depformer_logits_run {};
struct state_depformer_logits_sample_projection {};
struct state_depformer_logits_sample_projection_result_decision {};
struct state_depformer_logits_sample_select {};
struct state_depformer_token_publish {};
struct state_depformer_logits_result_decision {};
struct state_depformer_codebook_advance {};
struct state_step_error_out_decision {};
struct state_uninit_step_error_out_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using reset_run = event::reset_run;
    using init_run = event::initialize_run;
    using step_run = event::step_run;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Bind the Moshi LM execution contract.
        sml::state<state_bind_contract_decision> <= *sml::state<state_uninitialized>
          + sml::event<init_run>
      , sml::state<state_input_embedding_contract_decision> <=
          sml::state<state_bind_contract_decision>
          + sml::completion<init_run> [ guard::guard_bind_contract_valid{} ]
          / action::effect_bind_contract{}
      , sml::state<state_init_failed_error_out_decision> <= sml::state<state_bind_contract_decision>
          + sml::completion<init_run> [ guard::guard_bind_contract_invalid{} ]
          / action::effect_mark_bind_failed{}
      , sml::state<state_temporal_projection_layout_decision> <=
          sml::state<state_input_embedding_contract_decision>
          + sml::completion<init_run>
          [ guard::guard_bound_root_operands_supported{} ]
      , sml::state<state_init_failed_error_out_decision> <=
          sml::state<state_input_embedding_contract_decision>
          + sml::completion<init_run>
          [ guard::guard_bound_root_operands_unsupported{} ]
          / action::effect_mark_bind_failed{}
      , sml::state<state_depformer_projection_layout_decision> <=
          sml::state<state_temporal_projection_layout_decision>
          + sml::completion<init_run>
          [ guard::guard_temporal_split_projection_layout_supported{} ]
          / action::effect_bind_temporal_split_projection_layout{}
      , sml::state<state_depformer_projection_layout_decision> <=
          sml::state<state_temporal_projection_layout_decision>
          + sml::completion<init_run>
          [ guard::guard_temporal_fused_projection_layout_supported{} ]
          / action::effect_bind_temporal_fused_projection_layout{}
      , sml::state<state_init_failed_error_out_decision> <=
          sml::state<state_temporal_projection_layout_decision>
          + sml::completion<init_run>
          [ guard::guard_temporal_projection_layout_unsupported{} ]
          / action::effect_mark_bind_failed{}
      , sml::state<state_sampling_seed_decision> <=
          sml::state<state_depformer_projection_layout_decision>
          + sml::completion<init_run>
          [ guard::guard_depformer_split_projection_layout_supported{} ]
          / action::effect_bind_depformer_split_projection_layout{}
      , sml::state<state_sampling_seed_decision> <=
          sml::state<state_depformer_projection_layout_decision>
          + sml::completion<init_run>
          [ guard::guard_depformer_fused_projection_layout_supported{} ]
          / action::effect_bind_depformer_fused_projection_layout{}
      , sml::state<state_init_failed_error_out_decision> <=
          sml::state<state_depformer_projection_layout_decision>
          + sml::completion<init_run>
          [ guard::guard_depformer_projection_layout_unsupported{} ]
          / action::effect_mark_bind_failed{}
      , sml::state<state_text_sampling_top_k_decision> <= sml::state<state_sampling_seed_decision>
          + sml::completion<init_run> [ guard::guard_sampling_seed_nonzero{} ]
          / action::effect_bind_nonzero_sampling_seed{}
      , sml::state<state_text_sampling_top_k_decision> <= sml::state<state_sampling_seed_decision>
          + sml::completion<init_run> [ guard::guard_sampling_seed_zero{} ]
          / action::effect_bind_zero_sampling_seed{}
      , sml::state<state_audio_sampling_top_k_decision> <= sml::state<state_text_sampling_top_k_decision>
          + sml::completion<init_run> [ guard::guard_text_sampling_top_k_within_card{} ]
          / action::effect_bind_requested_text_sampling_top_k{}
      , sml::state<state_audio_sampling_top_k_decision> <= sml::state<state_text_sampling_top_k_decision>
          + sml::completion<init_run> [ guard::guard_text_sampling_top_k_exceeds_card{} ]
          / action::effect_bind_full_card_text_sampling_top_k{}
      , sml::state<state_init_error_out_decision> <= sml::state<state_audio_sampling_top_k_decision>
          + sml::completion<init_run> [ guard::guard_audio_sampling_top_k_within_card{} ]
          / action::effect_bind_requested_audio_sampling_top_k{}
      , sml::state<state_init_error_out_decision> <= sml::state<state_audio_sampling_top_k_decision>
          + sml::completion<init_run> [ guard::guard_audio_sampling_top_k_exceeds_card{} ]
          / action::effect_bind_full_card_audio_sampling_top_k{}
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
      , sml::state<state_ready> <= sml::state<state_init_callback_decision>
          + sml::completion<init_run> [ guard::guard_has_initialize_done_callback{} ]
          / action::effect_emit_initialize_done{}
      , sml::state<state_ready> <= sml::state<state_init_callback_decision>
          + sml::completion<init_run> [ guard::guard_no_initialize_done_callback{} ]
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_callback_decision>
          + sml::completion<init_run> [ guard::guard_has_initialize_error_callback{} ]
          / action::effect_emit_initialize_error{}
      , sml::state<state_uninitialized> <= sml::state<state_init_failed_callback_decision>
          + sml::completion<init_run> [ guard::guard_no_initialize_error_callback{} ]

      //------------------------------------------------------------------------------//
      // Graph-step request validation and Moshi LM execution.
      , sml::state<state_step_model_decision> <= sml::state<state_ready>
          + sml::event<step_run>
      , sml::state<state_step_shape_decision> <= sml::state<state_step_model_decision>
          + sml::completion<step_run> [ guard::guard_step_model_matches{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_step_model_decision>
          + sml::completion<step_run> [ guard::guard_step_model_mismatch{} ]
          / action::effect_mark_model_mismatch{}
      , sml::state<state_step_execution_decision> <= sml::state<state_step_shape_decision>
          + sml::completion<step_run> [ guard::guard_step_shape_valid{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_step_shape_decision>
          + sml::completion<step_run> [ guard::guard_step_shape_invalid{} ]
          / action::effect_mark_request_shape{}
      , sml::state<state_input_embedding> <= sml::state<state_step_execution_decision>
          + sml::completion<step_run> [ guard::guard_input_embedding_supported{} ]
          / action::effect_begin_input_embedding{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_step_execution_decision>
          + sml::completion<step_run> [ guard::guard_input_embedding_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_embedding_result_decision> <= sml::state<state_input_embedding>
          + sml::completion<step_run> [ guard::guard_external_input_embedding_supported{} ]
          / action::effect_apply_external_input_embedding{}
      , sml::state<state_input_text_embedding_bind> <= sml::state<state_input_embedding>
          + sml::completion<step_run> [ guard::guard_token_input_embedding_supported{} ]
          / action::effect_bind_input_text_embedding{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_embedding>
          + sml::completion<step_run> [ guard::guard_token_input_embedding_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_text_embedding_bind_result_decision> <= sml::state<state_input_text_embedding_bind>
          + sml::completion<step_run>
      , sml::state<state_input_text_embedding_row> <= sml::state<state_input_text_embedding_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_view_bound{} ]
          / action::effect_run_embedding_row_fetch{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_text_embedding_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_text_embedding_row_result_decision> <= sml::state<state_input_text_embedding_row>
          + sml::completion<step_run>
      , sml::state<state_input_text_embedding_apply> <= sml::state<state_input_text_embedding_row_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_row_succeeded{} ]
          / action::effect_apply_input_text_embedding_row{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_text_embedding_row_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_row_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_text_embedding_result_decision> <= sml::state<state_input_text_embedding_apply>
          + sml::completion<step_run> [ guard::guard_input_text_embedding_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_text_embedding_apply>
          + sml::completion<step_run> [ guard::guard_input_text_embedding_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_audio_token_decision> <= sml::state<state_input_text_embedding_result_decision>
          + sml::completion<step_run>
      , sml::state<state_input_audio_embedding> <= sml::state<state_input_audio_token_decision>
          + sml::completion<step_run> [ guard::guard_current_input_audio_token_present{} ]
          / action::effect_bind_input_audio_embedding{}
      , sml::state<state_input_audio_embedding_result_decision> <= sml::state<state_input_audio_token_decision>
          + sml::completion<step_run> [ guard::guard_current_input_audio_token_zero{} ]
          / action::effect_skip_zero_input_audio_embedding{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_audio_token_decision>
          + sml::completion<step_run> [ guard::guard_current_input_audio_token_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_audio_embedding_bind_result_decision> <= sml::state<state_input_audio_embedding>
          + sml::completion<step_run>
      , sml::state<state_input_audio_embedding_row> <= sml::state<state_input_audio_embedding_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_view_bound{} ]
          / action::effect_run_embedding_row_fetch{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_audio_embedding_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_audio_embedding_row_result_decision> <= sml::state<state_input_audio_embedding_row>
          + sml::completion<step_run>
      , sml::state<state_input_audio_embedding_apply> <= sml::state<state_input_audio_embedding_row_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_row_succeeded{} ]
          / action::effect_apply_input_audio_embedding_row{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_audio_embedding_row_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_row_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_audio_embedding_result_decision> <= sml::state<state_input_audio_embedding_apply>
          + sml::completion<step_run> [ guard::guard_input_audio_embedding_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_audio_embedding_apply>
          + sml::completion<step_run> [ guard::guard_input_audio_embedding_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_input_audio_codebook_advance> <= sml::state<state_input_audio_embedding_result_decision>
          + sml::completion<step_run> [ guard::guard_more_input_audio_codebooks{} ]
          / action::effect_advance_input_audio_codebook{}
      , sml::state<state_input_audio_token_decision> <= sml::state<state_input_audio_codebook_advance>
          + sml::completion<step_run>
      , sml::state<state_input_embedding_result_decision> <= sml::state<state_input_audio_embedding_result_decision>
          + sml::completion<step_run> [ guard::guard_input_audio_codebooks_complete{} ]
          / action::effect_finish_input_embedding{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_embedding_result_decision>
          + sml::completion<step_run> [ guard::guard_input_embedding_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_bind_temporal_kv> <= sml::state<state_input_embedding_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_kv_binding_present{} ]
          / action::effect_bind_temporal_kv{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_input_embedding_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_kv_binding_missing{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_kv_result_decision> <= sml::state<state_bind_temporal_kv>
          + sml::completion<step_run> [ guard::guard_temporal_kv_bound{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_bind_temporal_kv>
          + sml::completion<step_run> [ guard::guard_temporal_kv_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_position_result_decision> <=
            sml::state<state_temporal_kv_result_decision>
          + sml::completion<step_run> / action::effect_advance_temporal_position{}
      , sml::state<state_temporal_position_valid_decision> <=
            sml::state<state_temporal_position_result_decision>
          + sml::completion<step_run>
          [ guard::guard_temporal_position_advance_succeeded{} ]
      , sml::state<state_step_error_out_decision> <=
            sml::state<state_temporal_position_result_decision>
          + sml::completion<step_run>
          [ guard::guard_temporal_position_advance_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm> <=
            sml::state<state_temporal_position_valid_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_supported{} ]
          / action::effect_run_temporal_layer_norm_rms{}
      , sml::state<state_step_error_out_decision> <=
            sml::state<state_temporal_position_valid_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm_rms_result_decision> <= sml::state<state_temporal_layer_norm>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_norm_scale> <= sml::state<state_temporal_layer_norm_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_rms_succeeded{} ]
          / action::effect_run_temporal_layer_norm_scale{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_norm_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_rms_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm_result_decision> <= sml::state<state_temporal_layer_norm_scale>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_norm_scale>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_projection> <= sml::state<state_temporal_layer_norm_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_projection_supported{} ]
          / action::effect_bind_temporal_layer_projection{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_norm_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_projection_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_projection_bind_result_decision> <= sml::state<state_temporal_layer_projection>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_projection_run> <= sml::state<state_temporal_layer_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_temporal_layer_projection_serial{}
      , sml::state<state_temporal_layer_projection_run> <= sml::state<state_temporal_layer_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_temporal_layer_projection_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_projection_result_decision> <= sml::state<state_temporal_layer_projection_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_projection_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_projection_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_projection_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_rope> <= sml::state<state_temporal_layer_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_rope_supported{} ]
          / action::effect_run_temporal_layer_query_rope{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_rope_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_rope_result_decision> <= sml::state<state_temporal_layer_rope>
          + sml::completion<step_run> [ guard::guard_temporal_layer_rope_succeeded{} ]
          / action::effect_copy_temporal_layer_query_rope{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_rope>
          + sml::completion<step_run> [ guard::guard_temporal_layer_rope_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_key_rope> <= sml::state<state_temporal_layer_rope_result_decision>
          + sml::completion<step_run>
          / action::effect_run_temporal_layer_key_rope{}
      , sml::state<state_temporal_layer_key_rope_result_decision> <= sml::state<state_temporal_layer_key_rope>
          + sml::completion<step_run> [ guard::guard_temporal_layer_rope_succeeded{} ]
          / action::effect_copy_temporal_layer_key_rope{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_key_rope>
          + sml::completion<step_run> [ guard::guard_temporal_layer_rope_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_cache_write> <= sml::state<state_temporal_layer_key_rope_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_cache_write_supported{} ]
          / action::effect_write_temporal_layer_kv_cache{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_key_rope_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_cache_write_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_cache_write_result_decision> <= sml::state<state_temporal_layer_cache_write>
          + sml::completion<step_run> [ guard::guard_temporal_layer_cache_write_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_cache_write>
          + sml::completion<step_run> [ guard::guard_temporal_layer_cache_write_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_attention> <= sml::state<state_temporal_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_serial{} ]
          / action::effect_run_temporal_layer_attention_serial{}
      , sml::state<state_temporal_layer_attention> <= sml::state<state_temporal_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_two{} ]
          / action::effect_run_temporal_layer_attention_two{}
      , sml::state<state_temporal_layer_attention> <= sml::state<state_temporal_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_four{} ]
          / action::effect_run_temporal_layer_attention_four{}
      , sml::state<state_temporal_layer_attention> <= sml::state<state_temporal_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_eight{} ]
          / action::effect_run_temporal_layer_attention_eight{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_route_unavailable{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_attention_result_decision> <= sml::state<state_temporal_layer_attention>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_attention>
          + sml::completion<step_run> [ guard::guard_temporal_layer_attention_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_out_projection> <= sml::state<state_temporal_layer_attention_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_out_projection_supported{} ]
          / action::effect_bind_temporal_layer_out_projection{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_attention_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_out_projection_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_out_projection_bind_result_decision> <= sml::state<state_temporal_layer_out_projection>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_out_projection_run> <= sml::state<state_temporal_layer_out_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_temporal_layer_out_projection_serial{}
      , sml::state<state_temporal_layer_out_projection_run> <= sml::state<state_temporal_layer_out_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_temporal_layer_out_projection_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_out_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_out_projection_result_decision> <= sml::state<state_temporal_layer_out_projection_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_out_projection_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_out_projection_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_out_projection_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_residual> <= sml::state<state_temporal_layer_out_projection_result_decision>
          + sml::completion<step_run>
          / action::effect_apply_temporal_layer_attention_residual{}
      , sml::state<state_temporal_layer_residual_result_decision> <= sml::state<state_temporal_layer_residual>
          + sml::completion<step_run> [ guard::guard_temporal_layer_residual_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_residual>
          + sml::completion<step_run> [ guard::guard_temporal_layer_residual_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm2> <= sml::state<state_temporal_layer_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm2_supported{} ]
          / action::effect_run_temporal_layer_norm2_rms{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm2_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm2_rms_result_decision> <= sml::state<state_temporal_layer_norm2>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_norm2_scale> <= sml::state<state_temporal_layer_norm2_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm2_rms_succeeded{} ]
          / action::effect_run_temporal_layer_norm2_scale{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_norm2_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm2_rms_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm2_result_decision> <= sml::state<state_temporal_layer_norm2_scale>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm2_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_norm2_scale>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm2_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_gating_in> <= sml::state<state_temporal_layer_norm2_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_in_supported{} ]
          / action::effect_bind_temporal_layer_gating_in{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_norm2_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_in_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_gating_in_bind_result_decision> <= sml::state<state_temporal_layer_gating_in>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_gating_in_run> <= sml::state<state_temporal_layer_gating_in_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_temporal_layer_gating_in_serial{}
      , sml::state<state_temporal_layer_gating_in_run> <= sml::state<state_temporal_layer_gating_in_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_temporal_layer_gating_in_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_gating_in_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_gating_in_result_decision> <= sml::state<state_temporal_layer_gating_in_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_in_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_gating_in_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_in_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_silu_gate> <= sml::state<state_temporal_layer_gating_in_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_silu_gate_supported{} ]
          / action::effect_run_temporal_layer_silu_gate_silu{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_gating_in_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_silu_gate_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_silu_gate_silu_result_decision> <= sml::state<state_temporal_layer_silu_gate>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_silu_gate_mul> <= sml::state<state_temporal_layer_silu_gate_silu_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_silu_gate_silu_succeeded{} ]
          / action::effect_run_temporal_layer_silu_gate_mul{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_silu_gate_silu_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_silu_gate_silu_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_silu_gate_result_decision> <= sml::state<state_temporal_layer_silu_gate_mul>
          + sml::completion<step_run> [ guard::guard_temporal_layer_silu_gate_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_silu_gate_mul>
          + sml::completion<step_run> [ guard::guard_temporal_layer_silu_gate_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_gating_out> <= sml::state<state_temporal_layer_silu_gate_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_out_supported{} ]
          / action::effect_bind_temporal_layer_gating_out{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_silu_gate_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_out_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_gating_out_bind_result_decision> <= sml::state<state_temporal_layer_gating_out>
          + sml::completion<step_run>
      , sml::state<state_temporal_layer_gating_out_run> <= sml::state<state_temporal_layer_gating_out_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_temporal_layer_gating_out_serial{}
      , sml::state<state_temporal_layer_gating_out_run> <= sml::state<state_temporal_layer_gating_out_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_temporal_layer_gating_out_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_gating_out_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_gating_out_result_decision> <= sml::state<state_temporal_layer_gating_out_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_out_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_gating_out_run>
          + sml::completion<step_run> [ guard::guard_temporal_layer_gating_out_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_ff_residual> <= sml::state<state_temporal_layer_gating_out_result_decision>
          + sml::completion<step_run>
          / action::effect_apply_temporal_layer_ff_residual{}
      , sml::state<state_temporal_layer_ff_residual_result_decision> <= sml::state<state_temporal_layer_ff_residual>
          + sml::completion<step_run> [ guard::guard_temporal_layer_ff_residual_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_ff_residual>
          + sml::completion<step_run> [ guard::guard_temporal_layer_ff_residual_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_advance> <= sml::state<state_temporal_layer_ff_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_more_temporal_layers{} ]
          / action::effect_advance_temporal_layer{}
      , sml::state<state_temporal_out_norm> <= sml::state<state_temporal_layer_ff_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_layers_complete{} ]
          / action::effect_run_temporal_out_norm_rms{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_ff_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_out_norm_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_layer_norm> <= sml::state<state_temporal_layer_advance>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_supported{} ]
          / action::effect_run_temporal_layer_norm_rms{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_layer_advance>
          + sml::completion<step_run> [ guard::guard_temporal_layer_norm_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_out_norm_rms_result_decision> <= sml::state<state_temporal_out_norm>
          + sml::completion<step_run>
      , sml::state<state_temporal_out_norm_scale> <= sml::state<state_temporal_out_norm_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_out_norm_rms_succeeded{} ]
          / action::effect_run_temporal_out_norm_scale{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_out_norm_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_temporal_out_norm_rms_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_temporal_out_norm_result_decision> <= sml::state<state_temporal_out_norm_scale>
          + sml::completion<step_run> [ guard::guard_temporal_out_norm_succeeded{} ]
          / action::effect_publish_temporal_out_norm{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_temporal_out_norm_scale>
          + sml::completion<step_run> [ guard::guard_temporal_out_norm_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_text_logits_phase_decision> <=
          sml::state<state_temporal_out_norm_result_decision>
          + sml::completion<step_run> [ guard::guard_full_graph_phase{} ]
      , sml::state<state_sampling_ready> <=
          sml::state<state_temporal_out_norm_result_decision>
          + sml::completion<step_run> [ guard::guard_prediction_graph_phase{} ]
          / action::effect_publish_prediction_state{}
      , sml::state<state_text_logits> <= sml::state<state_text_logits_phase_decision>
          + sml::completion<step_run> [ guard::guard_text_logits_supported{} ]
          / action::effect_bind_text_token_logits{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_phase_decision>
          + sml::completion<step_run> [ guard::guard_text_logits_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_text_logits> <= sml::state<state_sampling_ready>
          + sml::event<step_run> [ guard::guard_sampling_step_valid{} ]
          / action::effect_restore_prediction_state_and_bind_text_logits{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_sampling_ready>
          + sml::event<step_run> [ guard::guard_sampling_step_invalid{} ]
          / action::effect_mark_request_shape{}
      , sml::state<state_text_logits_bind_result_decision> <= sml::state<state_text_logits>
          + sml::completion<step_run>
      , sml::state<state_text_logits_sample_projection> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_forced_text_token_sampling_serial{} ]
          / action::effect_compute_text_token_logits_serial{}
      , sml::state<state_text_logits_sample_projection> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_forced_text_token_sampling_parallel{} ]
          / action::effect_compute_text_token_logits_parallel{}
      , sml::state<state_text_logits_result_decision> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_forced_text_token_valid_without_sampling_consume{} ]
          / action::effect_publish_forced_text_token{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_forced_text_sampling_config_invalid{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_forced_text_token_invalid{} ]
          / action::effect_mark_request_shape{}
      , sml::state<state_text_logits_run> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_text_logits_projection_bound_and_no_forced_token_argmax{} ]
          / action::effect_select_text_token{}
      , sml::state<state_text_logits_sample_projection> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_text_logits_sampling_serial{} ]
          / action::effect_compute_text_token_logits_serial{}
      , sml::state<state_text_logits_sample_projection> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_text_logits_sampling_parallel{} ]
          / action::effect_compute_text_token_logits_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_text_logits_projection_bound_and_no_forced_token_sampling_invalid{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_text_logits_sample_projection_result_decision> <= sml::state<state_text_logits_sample_projection>
          + sml::completion<step_run>
      , sml::state<state_text_logits_sample_select> <= sml::state<state_text_logits_sample_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_text_logits_matmul_succeeded{} ]
          / action::effect_select_text_sampling_token{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_sample_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_text_logits_matmul_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_text_logits_result_decision> <= sml::state<state_text_logits_sample_select>
          + sml::completion<step_run> [ guard::guard_sampled_text_token_ready{} ]
          / action::effect_publish_sampled_text_token{}
      , sml::state<state_text_logits_result_decision> <= sml::state<state_text_logits_sample_select>
          + sml::completion<step_run> [ guard::guard_forced_text_sampling_consumed{} ]
          / action::effect_publish_forced_text_token{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_sample_select>
          + sml::completion<step_run> [ guard::guard_text_sampling_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_text_logits_result_decision> <= sml::state<state_text_logits_run>
          + sml::completion<step_run> [ guard::guard_text_logits_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_run>
          + sml::completion<step_run> [ guard::guard_text_logits_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_bind_depformer_kv> <= sml::state<state_text_logits_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_kv_binding_present{} ]
          / action::effect_bind_depformer_kv{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_text_logits_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_kv_binding_missing{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_kv_result_decision> <= sml::state<state_bind_depformer_kv>
          + sml::completion<step_run> [ guard::guard_depformer_kv_bound{} ]
          / action::effect_reset_depformer_positions{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_bind_depformer_kv>
          + sml::completion<step_run> [ guard::guard_depformer_kv_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_position_reset_result_decision> <=
            sml::state<state_depformer_kv_result_decision>
          + sml::completion<step_run>
      , sml::state<state_depformer_position_advance_result_decision> <=
            sml::state<state_depformer_position_reset_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_position_reset_succeeded{} ]
          / action::effect_advance_depformer_position{}
      , sml::state<state_step_error_out_decision> <=
            sml::state<state_depformer_position_reset_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_position_reset_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_weight_route_decision> <=
            sml::state<state_depformer_position_advance_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_position_advance_succeeded{} ]
      , sml::state<state_step_error_out_decision> <=
            sml::state<state_depformer_position_advance_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_position_advance_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_weight_decision> <=
            sml::state<state_depformer_weight_route_decision>
          + sml::completion<step_run> [ guard::guard_depformer_scheduled_weight_present{} ]
          / action::effect_use_depformer_scheduled_weight{}
      , sml::state<state_depformer_weight_decision> <=
            sml::state<state_depformer_weight_route_decision>
          + sml::completion<step_run> [ guard::guard_depformer_scheduled_weight_absent{} ]
          / action::effect_use_depformer_codebook_weight{}
      , sml::state<state_step_error_out_decision> <=
            sml::state<state_depformer_weight_route_decision>
          + sml::completion<step_run> [ guard::guard_depformer_scheduled_weight_invalid{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_input> <= sml::state<state_depformer_weight_decision>
          + sml::completion<step_run> [ guard::guard_depformer_text_input_supported{} ]
          / action::effect_bind_depformer_text_input_projection{}
      , sml::state<state_depformer_input> <= sml::state<state_depformer_weight_decision>
          + sml::completion<step_run> [ guard::guard_depformer_audio_input_supported{} ]
          / action::effect_bind_depformer_audio_input_projection{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_weight_decision>
          + sml::completion<step_run> [ guard::guard_depformer_input_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_input_projection_bind_result_decision> <= sml::state<state_depformer_input>
          + sml::completion<step_run>
      , sml::state<state_depformer_input_projection> <= sml::state<state_depformer_input_projection_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_input_projection_serial{} ]
          / action::effect_run_depformer_input_projection_serial{}
      , sml::state<state_depformer_input_projection> <= sml::state<state_depformer_input_projection_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_input_projection_parallel{} ]
          / action::effect_run_depformer_input_projection_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_input_projection_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_input_projection_result_decision> <= sml::state<state_depformer_input_projection>
          + sml::completion<step_run>
      , sml::state<state_depformer_input_embedding_bind> <= sml::state<state_depformer_input_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_text_input_projection_succeeded{} ]
          / action::effect_bind_depformer_text_input_embedding{}
      , sml::state<state_depformer_input_embedding_bind> <= sml::state<state_depformer_input_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_audio_input_projection_succeeded{} ]
          / action::effect_bind_depformer_audio_input_embedding{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_input_projection_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_input_projection_embedding_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_input_embedding_bind_result_decision> <= sml::state<state_depformer_input_embedding_bind>
          + sml::completion<step_run>
      , sml::state<state_depformer_input_embedding_row> <= sml::state<state_depformer_input_embedding_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_view_bound{} ]
          / action::effect_run_embedding_row_fetch{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_embedding_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_input_embedding_row_result_decision> <= sml::state<state_depformer_input_embedding_row>
          + sml::completion<step_run>
      , sml::state<state_depformer_input_embedding_apply> <= sml::state<state_depformer_input_embedding_row_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_row_succeeded{} ]
          / action::effect_apply_depformer_input_embedding_row{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_embedding_row_result_decision>
          + sml::completion<step_run> [ guard::guard_embedding_row_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_input_result_decision> <= sml::state<state_depformer_input_embedding_apply>
          + sml::completion<step_run> [ guard::guard_depformer_input_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_embedding_apply>
          + sml::completion<step_run> [ guard::guard_depformer_input_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm> <= sml::state<state_depformer_input_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_supported{} ]
          / action::effect_run_depformer_layer_norm_rms{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_input_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm_rms_result_decision> <= sml::state<state_depformer_layer_norm>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_norm_scale> <= sml::state<state_depformer_layer_norm_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_rms_succeeded{} ]
          / action::effect_run_depformer_layer_norm_scale{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_norm_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_rms_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm_result_decision> <= sml::state<state_depformer_layer_norm_scale>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_norm_scale>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_projection> <= sml::state<state_depformer_layer_norm_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_projection_supported{} ]
          / action::effect_bind_depformer_layer_projection{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_norm_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_projection_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_projection_bind_result_decision> <= sml::state<state_depformer_layer_projection>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_projection_run> <= sml::state<state_depformer_layer_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_depformer_layer_projection_serial{}
      , sml::state<state_depformer_layer_projection_run> <= sml::state<state_depformer_layer_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_depformer_layer_projection_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_projection_result_decision> <= sml::state<state_depformer_layer_projection_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_projection_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_projection_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_projection_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_cache_write> <= sml::state<state_depformer_layer_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_cache_write_supported{} ]
          / action::effect_write_depformer_layer_kv_cache{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_cache_write_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_cache_write_result_decision> <= sml::state<state_depformer_layer_cache_write>
          + sml::completion<step_run> [ guard::guard_depformer_layer_cache_write_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_cache_write>
          + sml::completion<step_run> [ guard::guard_depformer_layer_cache_write_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_attention> <= sml::state<state_depformer_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_attention_supported{} ]
          / action::effect_run_depformer_layer_attention{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_cache_write_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_attention_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_attention_result_decision> <= sml::state<state_depformer_layer_attention>
          + sml::completion<step_run> [ guard::guard_depformer_layer_attention_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_attention>
          + sml::completion<step_run> [ guard::guard_depformer_layer_attention_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_out_projection> <= sml::state<state_depformer_layer_attention_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_out_projection_supported{} ]
          / action::effect_bind_depformer_layer_out_projection{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_attention_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_out_projection_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_out_projection_bind_result_decision> <= sml::state<state_depformer_layer_out_projection>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_out_projection_run> <= sml::state<state_depformer_layer_out_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_depformer_layer_out_projection_serial{}
      , sml::state<state_depformer_layer_out_projection_run> <= sml::state<state_depformer_layer_out_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_depformer_layer_out_projection_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_out_projection_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_out_projection_result_decision> <= sml::state<state_depformer_layer_out_projection_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_out_projection_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_out_projection_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_out_projection_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_residual> <= sml::state<state_depformer_layer_out_projection_result_decision>
          + sml::completion<step_run>
          / action::effect_apply_depformer_layer_attention_residual{}
      , sml::state<state_depformer_layer_residual_result_decision> <= sml::state<state_depformer_layer_residual>
          + sml::completion<step_run> [ guard::guard_depformer_layer_residual_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_residual>
          + sml::completion<step_run> [ guard::guard_depformer_layer_residual_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm2> <= sml::state<state_depformer_layer_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm2_supported{} ]
          / action::effect_run_depformer_layer_norm2_rms{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm2_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm2_rms_result_decision> <= sml::state<state_depformer_layer_norm2>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_norm2_scale> <= sml::state<state_depformer_layer_norm2_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm2_rms_succeeded{} ]
          / action::effect_run_depformer_layer_norm2_scale{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_norm2_rms_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm2_rms_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm2_result_decision> <= sml::state<state_depformer_layer_norm2_scale>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm2_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_norm2_scale>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm2_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_gating_in> <= sml::state<state_depformer_layer_norm2_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_in_supported{} ]
          / action::effect_bind_depformer_layer_gating_in{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_norm2_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_in_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_gating_in_bind_result_decision> <= sml::state<state_depformer_layer_gating_in>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_gating_in_run> <= sml::state<state_depformer_layer_gating_in_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_depformer_layer_gating_in_serial{}
      , sml::state<state_depformer_layer_gating_in_run> <= sml::state<state_depformer_layer_gating_in_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_depformer_layer_gating_in_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_gating_in_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_gating_in_result_decision> <= sml::state<state_depformer_layer_gating_in_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_in_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_gating_in_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_in_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_silu_gate> <= sml::state<state_depformer_layer_gating_in_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_silu_gate_supported{} ]
          / action::effect_run_depformer_layer_silu_gate_silu{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_gating_in_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_silu_gate_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_silu_gate_silu_result_decision> <= sml::state<state_depformer_layer_silu_gate>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_silu_gate_mul> <= sml::state<state_depformer_layer_silu_gate_silu_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_silu_gate_silu_succeeded{} ]
          / action::effect_run_depformer_layer_silu_gate_mul{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_silu_gate_silu_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_silu_gate_silu_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_silu_gate_result_decision> <= sml::state<state_depformer_layer_silu_gate_mul>
          + sml::completion<step_run> [ guard::guard_depformer_layer_silu_gate_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_silu_gate_mul>
          + sml::completion<step_run> [ guard::guard_depformer_layer_silu_gate_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_gating_out> <= sml::state<state_depformer_layer_silu_gate_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_out_supported{} ]
          / action::effect_bind_depformer_layer_gating_out{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_silu_gate_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_out_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_gating_out_bind_result_decision> <= sml::state<state_depformer_layer_gating_out>
          + sml::completion<step_run>
      , sml::state<state_depformer_layer_gating_out_run> <= sml::state<state_depformer_layer_gating_out_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_serial{} ]
          / action::effect_run_depformer_layer_gating_out_serial{}
      , sml::state<state_depformer_layer_gating_out_run> <= sml::state<state_depformer_layer_gating_out_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bound_parallel{} ]
          / action::effect_run_depformer_layer_gating_out_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_gating_out_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_gating_out_result_decision> <= sml::state<state_depformer_layer_gating_out_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_out_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_gating_out_run>
          + sml::completion<step_run> [ guard::guard_depformer_layer_gating_out_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_ff_residual> <= sml::state<state_depformer_layer_gating_out_result_decision>
          + sml::completion<step_run>
          / action::effect_apply_depformer_layer_ff_residual{}
      , sml::state<state_depformer_layer_ff_residual_result_decision> <= sml::state<state_depformer_layer_ff_residual>
          + sml::completion<step_run> [ guard::guard_depformer_layer_ff_residual_succeeded{} ]
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_ff_residual>
          + sml::completion<step_run> [ guard::guard_depformer_layer_ff_residual_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_advance> <= sml::state<state_depformer_layer_ff_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_more_depformer_layers{} ]
          / action::effect_advance_depformer_layer{}
      , sml::state<state_depformer_logits> <= sml::state<state_depformer_layer_ff_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_logits_supported{} ]
          / action::effect_bind_depformer_token_logits{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_ff_residual_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_logits_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_logits_bind_result_decision> <= sml::state<state_depformer_logits>
          + sml::completion<step_run>
      , sml::state<state_depformer_logits_run> <= sml::state<state_depformer_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_logits_projection_bound_argmax{} ]
          / action::effect_select_depformer_token{}
      , sml::state<state_depformer_logits_sample_projection> <= sml::state<state_depformer_logits_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_logits_sampling_serial{} ]
          / action::effect_compute_depformer_token_logits_serial{}
      , sml::state<state_depformer_logits_sample_projection> <= sml::state<state_depformer_logits_bind_result_decision>
          + sml::completion<step_run>
          [ guard::guard_depformer_logits_sampling_parallel{} ]
          / action::effect_compute_depformer_token_logits_parallel{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_logits_projection_bound_sampling_invalid{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_logits_bind_result_decision>
          + sml::completion<step_run> [ guard::guard_projection_view_bind_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_logits_sample_projection_result_decision> <= sml::state<state_depformer_logits_sample_projection>
          + sml::completion<step_run>
      , sml::state<state_depformer_logits_sample_select> <= sml::state<state_depformer_logits_sample_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_logits_matmul_succeeded{} ]
          / action::effect_select_depformer_sampling_token{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_logits_sample_projection_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_logits_matmul_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_layer_norm> <= sml::state<state_depformer_layer_advance>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_supported{} ]
          / action::effect_run_depformer_layer_norm_rms{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_layer_advance>
          + sml::completion<step_run> [ guard::guard_depformer_layer_norm_unsupported{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_token_publish> <= sml::state<state_depformer_logits_run>
          + sml::completion<step_run> [ guard::guard_depformer_logits_succeeded{} ]
          / action::effect_publish_depformer_token{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_logits_run>
          + sml::completion<step_run> [ guard::guard_depformer_logits_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_token_publish> <= sml::state<state_depformer_logits_sample_select>
          + sml::completion<step_run> [ guard::guard_depformer_sampling_succeeded{} ]
          / action::effect_publish_depformer_token{}
      , sml::state<state_step_error_out_decision> <= sml::state<state_depformer_logits_sample_select>
          + sml::completion<step_run> [ guard::guard_depformer_sampling_failed{} ]
          / action::effect_mark_graph_execution_unsupported{}
      , sml::state<state_depformer_logits_result_decision> <= sml::state<state_depformer_token_publish>
          + sml::completion<step_run>
      , sml::state<state_depformer_codebook_advance> <= sml::state<state_depformer_logits_result_decision>
          + sml::completion<step_run> [ guard::guard_more_depformer_codebooks{} ]
          / action::effect_advance_depformer_codebook{}
      , sml::state<state_ready> <= sml::state<state_depformer_logits_result_decision>
          + sml::completion<step_run> [ guard::guard_depformer_codebooks_complete{} ]
      , sml::state<state_depformer_position_advance_result_decision> <=
            sml::state<state_depformer_codebook_advance>
          + sml::completion<step_run> / action::effect_advance_depformer_position{}
      , sml::state<state_ready> <= sml::state<state_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_has_error_out<step_run>{} ]
          / action::effect_store_error_out<step_run>{}
      , sml::state<state_ready> <= sml::state<state_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_no_error_out<step_run>{} ]

      //------------------------------------------------------------------------------//
      // Requests before initialization answer explicitly.
      , sml::state<state_uninit_step_error_out_decision> <= sml::state<state_uninitialized>
          + sml::event<step_run>
          / action::effect_mark_not_initialized{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_has_error_out<step_run>{} ]
          / action::effect_store_error_out<step_run>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninit_step_error_out_decision>
          + sml::completion<step_run> [ guard::guard_no_error_out<step_run>{} ]

      //------------------------------------------------------------------------------//
      // Reset.
      , sml::state<state_reset_temporal_positions_decision> <= sml::state<state_ready>
          + sml::event<reset_run>
      , sml::state<state_reset_temporal_positions_decision> <= sml::state<state_sampling_ready>
          + sml::event<reset_run>
      , sml::state<state_reset_temporal_positions_result_decision> <=
          sml::state<state_reset_temporal_positions_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_temporal_positions_present{} ]
          / action::effect_reset_temporal_positions{}
      , sml::state<state_reset_depformer_positions_decision> <=
          sml::state<state_reset_temporal_positions_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_temporal_positions_missing{} ]
      , sml::state<state_reset_depformer_positions_decision> <=
          sml::state<state_reset_temporal_positions_result_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_temporal_positions_succeeded{} ]
      , sml::state<state_reset_failed> <=
          sml::state<state_reset_temporal_positions_result_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_temporal_positions_failed{} ]
          / action::effect_mark_reset_failed{}
      , sml::state<state_reset_depformer_positions_result_decision> <=
          sml::state<state_reset_depformer_positions_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_depformer_positions_present{} ]
          / action::effect_reset_bound_depformer_positions{}
      , sml::state<state_uninitialized> <=
          sml::state<state_reset_depformer_positions_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_depformer_positions_missing{} ]
          / action::effect_reset_session{}
      , sml::state<state_uninitialized> <=
          sml::state<state_reset_depformer_positions_result_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_depformer_positions_succeeded{} ]
          / action::effect_reset_session{}
      , sml::state<state_reset_failed> <=
          sml::state<state_reset_depformer_positions_result_decision>
          + sml::completion<reset_run>
              [ guard::guard_reset_depformer_positions_failed{} ]
          / action::effect_mark_reset_failed{}
      , sml::state<state_uninitialized> <= sml::state<state_reset_failed>
          + sml::completion<reset_run> / action::effect_reset_session{}

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
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_absent{} ]
              / action::effect_mark_unexpected{}
      , sml::state<state_sampling_ready> <= sml::state<state_sampling_ready>
          + sml::unexpected_event<sml::_>
              [ guard::guard_unexpected_error_out_present{} ]
              / action::effect_mark_unexpected_and_store{}
      , sml::state<state_sampling_ready> <= sml::state<state_sampling_ready>
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

  explicit sm(const action::dependencies &deps)
      : base_type(std::in_place, deps) {}

  bool process_event(const event::initialize &ev) {
    event::initialize_ctx ctx{};
    event::initialize_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(
      const emel::speech::predictor::moshi::event::initialize_graph &ev) {
    event::initialize request{ev.request.model};
    request.sampling_enabled = ev.request.sampling_enabled;
    request.sampling_consume_forced_text =
        ev.request.sampling_consume_forced_text;
    request.sampling_audio_temperature = ev.request.sampling_audio_temperature;
    request.sampling_text_temperature = ev.request.sampling_text_temperature;
    request.sampling_audio_top_k = ev.request.sampling_audio_top_k;
    request.sampling_text_top_k = ev.request.sampling_text_top_k;
    request.sampling_seed = ev.request.sampling_seed;
    request.error_out = ev.error_out;
    return process_event(request);
  }

  bool
  process_event(const emel::speech::predictor::moshi::event::graph_step &ev) {
    event::step_run runtime_ev{ev, step_workspace_};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted &&
           step_workspace_.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::reset &ev) {
    event::reset_ctx ctx{};
    event::reset_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const emel::speech::predictor::moshi::event::reset &) {
    return process_event(event::reset{});
  }

private:
  // Actor-owned numeric scratch is initialized once and reused synchronously.
  // Entry actions reset scalar outcomes and only the active tensor extents.
  event::step_ctx step_workspace_ = {};
};

} // namespace emel::speech::predictor::moshi::executor
