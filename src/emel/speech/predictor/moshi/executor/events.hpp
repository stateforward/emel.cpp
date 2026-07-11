#pragma once

#include <array>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/memory/streaming/events.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/predictor/moshi/events.hpp"
#include "emel/speech/predictor/moshi/executor/detail.hpp"
#include "emel/speech/predictor/moshi/executor/errors.hpp"

namespace emel::speech::predictor::moshi::executor::events {

struct initialize_done;
struct initialize_error;

} // namespace emel::speech::predictor::moshi::executor::events

namespace emel::speech::predictor::moshi::executor::event {

struct initialize {
  explicit initialize(const emel::model::data &model_ref) noexcept
      : model(model_ref) {}

  const emel::model::data &model;
  bool sampling_enabled = false;
  bool sampling_consume_forced_text = false;
  float sampling_audio_temperature = 0.0f;
  float sampling_text_temperature = 0.0f;
  int32_t sampling_audio_top_k = 0;
  int32_t sampling_text_top_k = 0;
  uint32_t sampling_seed = 1u;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct step_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool input_embedding_ok = false;
  bool input_text_embedding_ok = false;
  bool input_audio_embedding_ok = false;
  bool embedding_view_bound = false;
  bool embedding_row_ok = false;
  bool projection_view_bound = false;
  bool temporal_kv_bound = false;
  bool temporal_position_accepted = false;
  bool temporal_layer_norm_rms_ok = false;
  bool temporal_layer_norm_ok = false;
  bool temporal_layer_projection_ok = false;
  bool temporal_layer_rope_ok = false;
  bool temporal_layer_cache_write_ok = false;
  bool temporal_layer_attention_ok = false;
  bool temporal_layer_out_projection_ok = false;
  bool temporal_layer_residual_ok = false;
  bool temporal_layer_norm2_rms_ok = false;
  bool temporal_layer_norm2_ok = false;
  bool temporal_layer_gating_in_ok = false;
  bool temporal_layer_silu_gate_silu_ok = false;
  bool temporal_layer_silu_gate_ok = false;
  bool temporal_layer_gating_out_ok = false;
  bool temporal_layer_ff_residual_ok = false;
  bool temporal_out_norm_rms_ok = false;
  bool temporal_out_norm_ok = false;
  bool text_logits_ok = false;
  bool depformer_kv_bound = false;
  bool depformer_position_accepted = false;
  bool depformer_input_ok = false;
  bool depformer_input_projection_bound = false;
  bool depformer_input_projection_ok = false;
  bool depformer_layer_norm_rms_ok = false;
  bool depformer_layer_norm_ok = false;
  bool depformer_layer_projection_ok = false;
  bool depformer_layer_cache_write_ok = false;
  bool depformer_layer_attention_ok = false;
  bool depformer_layer_out_projection_ok = false;
  bool depformer_layer_residual_ok = false;
  bool depformer_layer_norm2_rms_ok = false;
  bool depformer_layer_norm2_ok = false;
  bool depformer_layer_gating_in_ok = false;
  bool depformer_layer_silu_gate_silu_ok = false;
  bool depformer_layer_silu_gate_ok = false;
  bool depformer_layer_gating_out_ok = false;
  bool depformer_layer_ff_residual_ok = false;
  bool depformer_logits_ok = false;
  bool sampler_accepted = false;
  int32_t row_index = 0;
  int32_t embedding_dim = 0;
  int32_t best_index = -1;
  int32_t input_audio_codebook_index = 0;
  int32_t temporal_layer_index = 0;
  int32_t temporal_rope_position = -1;
  int32_t temporal_position_error = 0;
  int32_t depformer_codebook_index = 0;
  int32_t depformer_weight_index = -1;
  int32_t depformer_layer_index = 0;
  int32_t depformer_position_error = 0;
  emel::error::type sampler_error = {};
  float best_score = 0.0f;
  detail::temporal_kv_view temporal_kv = {};
  detail::depformer_kv_view depformer_kv = {};
  emel::memory::streaming::advance_result temporal_position = {};
  emel::memory::streaming::advance_result depformer_position = {};
  detail::tensor_view embedding_view = {};
  detail::tensor_view projection_view = {};
  alignas(64) std::array<float, detail::k_max_sampling_card> logits = {};
  alignas(64) std::array<float, detail::k_max_sampling_top_k> top_scores = {};
  alignas(
      64) std::array<int32_t, detail::k_max_sampling_top_k> top_indices = {};
  alignas(64)
      std::array<int32_t, detail::k_max_sampling_card> sampling_indices = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> row = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> hidden = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> transformer_out = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> normalized = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> attention = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> projection = {};
  alignas(64) std::array<float, detail::k_max_qkv_dim> qkv = {};
  alignas(64) std::array<float, detail::k_max_hidden_dim> rope = {};
  alignas(64)
      std::array<float, detail::k_max_gating_projection_dim> gating_projection =
          {};
  alignas(64) std::array<float, detail::k_max_gating_dim> gating_silu = {};
  alignas(64) std::array<float, detail::k_max_gating_dim> gated = {};
  alignas(64)
      std::array<float, detail::k_max_temporal_context> attention_scores = {};
  alignas(64) std::array<uint16_t, detail::k_max_hidden_dim> q_bf16 = {};
  alignas(64) std::array<
      uint16_t, detail::k_max_temporal_context> attention_weights_bf16 = {};
};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};

struct step_run {
  const emel::speech::predictor::moshi::event::graph_step &request;
  step_ctx &ctx;
};

struct reset {};

} // namespace emel::speech::predictor::moshi::executor::event

namespace emel::speech::predictor::moshi::executor::events {

struct initialize_done {
  const event::initialize *request = nullptr;
  int32_t n_q = 0;
  int32_t dep_q = 0;
};

struct initialize_error {
  const event::initialize *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::predictor::moshi::executor::events
