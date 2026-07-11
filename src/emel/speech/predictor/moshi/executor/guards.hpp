#pragma once

#include <cmath>
#include <limits>

#include "emel/model/loader/errors.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/executor/context.hpp"
#include "emel/speech/predictor/moshi/executor/detail.hpp"
#include "emel/speech/predictor/moshi/executor/events.hpp"

namespace emel::speech::predictor::moshi::executor::guard {

inline bool guard_cache_span_valid(const size_t layer_offset,
                                   const size_t first_position,
                                   const size_t position_count,
                                   const size_t dim,
                                   const size_t key_cache_size,
                                   const size_t value_cache_size) noexcept {
  if (dim == 0u || first_position >
                       std::numeric_limits<size_t>::max() / dim ||
      position_count > std::numeric_limits<size_t>::max() / dim) {
    return false;
  }
  const size_t first_element = first_position * dim;
  const size_t element_count = position_count * dim;
  if (layer_offset >
      std::numeric_limits<size_t>::max() - first_element) {
    return false;
  }
  const size_t begin = layer_offset + first_element;
  return begin <= key_cache_size && begin <= value_cache_size &&
         element_count <= key_cache_size - begin &&
         element_count <= value_cache_size - begin;
}

struct guard_bind_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const bool sampling_config_valid =
        !runtime_ev.request.sampling_enabled ||
        (std::isfinite(runtime_ev.request.sampling_audio_temperature) &&
         runtime_ev.request.sampling_audio_temperature > 0.0f &&
         std::isfinite(runtime_ev.request.sampling_text_temperature) &&
         runtime_ev.request.sampling_text_temperature > 0.0f &&
         runtime_ev.request.sampling_audio_top_k > 0 &&
         runtime_ev.request.sampling_text_top_k > 0);
    return emel::model::moshi::detail::validate_execution_contract(model) ==
               emel::error::cast(emel::model::loader::error::none) &&
           model.moshi_component_id == emel::model::data::moshi_component::lm &&
           model.moshi_lm.n_q > 0 && model.moshi_lm.dep_q > 0 &&
           model.moshi_lm.dep_q <= model.moshi_lm.n_q &&
           model.moshi_lm.text_card > 0 && model.moshi_lm.card > 0 &&
           model.moshi_lm.dim > 0 &&
           (!runtime_ev.request.sampling_enabled || ctx.sampler != nullptr) &&
           sampling_config_valid &&
           ctx.policy.rms_norm_epsilon > 0.0f &&
           ctx.policy.sampling_modulus > 1u &&
           ctx.policy.zero_seed_state % ctx.policy.sampling_modulus != 0u &&
           ctx.policy.token_zero < 0 &&
           ctx.capacity.hidden_dim > 0u &&
           ctx.capacity.hidden_dim <= detail::k_max_hidden_dim &&
           ctx.capacity.temporal_context > 0u &&
           ctx.capacity.temporal_context <= detail::k_max_temporal_context &&
           ctx.capacity.depformer_context > 0u &&
           ctx.capacity.depformer_context <= detail::k_max_depformer_context &&
           ctx.capacity.sampling_card > 0u &&
           ctx.capacity.sampling_card <= detail::k_max_sampling_card &&
           ctx.capacity.sampling_top_k > 0u &&
           ctx.capacity.sampling_top_k <= detail::k_max_sampling_top_k;
  }
};

struct guard_bind_contract_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_bound_root_operands_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const auto *text_emb = ctx.session.contract.lm.text_embedding.tensor;
    const auto *output_norm = ctx.session.contract.lm.output_norm.tensor;
    const auto *text_output_projection =
        ctx.session.contract.lm.text_output_projection.tensor;
    bool supported =
        detail::tensor_shape(text_emb, hidden_dim, lm.text_card + 1) &&
        detail::supported_get_rows_dtype(
            static_cast<detail::dtype>(text_emb->type)) &&
        detail::tensor_shape(output_norm, hidden_dim) &&
        static_cast<detail::dtype>(output_norm->type) == detail::dtype::f32 &&
        detail::tensor_shape(text_output_projection, hidden_dim,
                             lm.text_card) &&
        detail::supported_argmax_dtype(
            static_cast<detail::dtype>(text_output_projection->type));
    for (int32_t index = 0; index < lm.n_q; ++index) {
      const auto *audio_emb =
          ctx.session.contract.lm.audio_embeddings[static_cast<size_t>(index)]
              .tensor;
      supported = supported &&
                  detail::tensor_shape(audio_emb, hidden_dim, lm.card + 1) &&
                  detail::supported_get_rows_dtype(
                      static_cast<detail::dtype>(audio_emb->type));
    }
    const int32_t dep_dim = lm.depformer_dim;
    const auto *depformer_text_embedding =
        ctx.session.contract.lm.depformer_text_embedding.tensor;
    supported = supported && dep_dim > 0 &&
                detail::tensor_shape(depformer_text_embedding, dep_dim,
                                     lm.text_card + 1) &&
                detail::supported_get_rows_dtype(
                    static_cast<detail::dtype>(depformer_text_embedding->type));
    for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
      const auto *input_projection =
          ctx.session.contract.lm
              .depformer_input_projections[static_cast<size_t>(codebook)]
              .tensor;
      const auto *output_projection =
          ctx.session.contract.lm
              .depformer_output_projections[static_cast<size_t>(codebook)]
              .tensor;
      supported = supported &&
                  detail::tensor_shape(input_projection, hidden_dim, dep_dim) &&
                  detail::supported_mul_mat_dtype(
                      static_cast<detail::dtype>(input_projection->type)) &&
                  detail::tensor_shape(output_projection, dep_dim, lm.card) &&
                  detail::supported_argmax_dtype(
                      static_cast<detail::dtype>(output_projection->type));
      if (codebook > 0) {
        const auto *audio_embedding =
            ctx.session.contract.lm
                .depformer_audio_embeddings[static_cast<size_t>(codebook - 1)]
                .tensor;
        supported =
            supported &&
            detail::tensor_shape(audio_embedding, dep_dim, lm.card + 1) &&
            detail::supported_get_rows_dtype(
                static_cast<detail::dtype>(audio_embedding->type));
      }
    }
    for (int32_t layer_index = 0; layer_index < lm.num_layers; ++layer_index) {
      const auto &layer =
          ctx.session.contract.lm
              .temporal_layers[static_cast<size_t>(layer_index)];
      const auto *norm1 = layer.norm1.tensor;
      const auto *output_projection = layer.output_projection.tensor;
      const auto *norm2 = layer.norm2.tensor;
      const auto *gating_input = layer.gating_input.tensor;
      const auto *gating_output = layer.gating_output.tensor;
      int64_t gating_projection_dim = 0;
      if (gating_input != nullptr && gating_input->n_dims > 1) {
        gating_projection_dim = gating_input->dims[1];
      }
      const int64_t gating_dim = gating_projection_dim / 2;
      supported =
          supported && detail::tensor_shape(norm1, hidden_dim) &&
          static_cast<detail::dtype>(norm1->type) == detail::dtype::f32 &&
          detail::tensor_shape(output_projection, hidden_dim, hidden_dim) &&
          detail::supported_mul_mat_dtype(
              static_cast<detail::dtype>(output_projection->type)) &&
          detail::tensor_shape(norm2, hidden_dim) &&
          static_cast<detail::dtype>(norm2->type) == detail::dtype::f32 &&
          gating_projection_dim > 0 && (gating_projection_dim % 2) == 0 &&
          static_cast<uint64_t>(gating_projection_dim) <=
              ctx.capacity.hidden_dim * 8u &&
          detail::tensor_shape(gating_input, hidden_dim,
                               gating_projection_dim) &&
          detail::supported_mul_mat_dtype(
              static_cast<detail::dtype>(gating_input->type)) &&
          gating_dim > 0 &&
          static_cast<uint64_t>(gating_dim) <= ctx.capacity.hidden_dim * 4u &&
          detail::tensor_shape(gating_output, gating_dim, hidden_dim) &&
          detail::supported_mul_mat_dtype(
              static_cast<detail::dtype>(gating_output->type));
    }
    for (int32_t layer_index = 0; layer_index < lm.depformer_num_layers;
         ++layer_index) {
      const auto &layer =
          ctx.session.contract.lm
              .depformer_layers[static_cast<size_t>(layer_index)];
      const auto *norm1 = layer.norm1.tensor;
      const auto *norm2 = layer.norm2.tensor;
      supported =
          supported && detail::tensor_shape(norm1, dep_dim) &&
          static_cast<detail::dtype>(norm1->type) == detail::dtype::f32 &&
          detail::tensor_shape(norm2, dep_dim) &&
          static_cast<detail::dtype>(norm2->type) == detail::dtype::f32;
      for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
        const auto &codebook_layer =
            layer.codebooks[static_cast<size_t>(codebook)];
        const auto *output_projection = codebook_layer.output_projection.tensor;
        const auto *gating_input = codebook_layer.gating_input.tensor;
        const auto *gating_output = codebook_layer.gating_output.tensor;
        int64_t gating_projection_dim = 0;
        if (gating_input != nullptr && gating_input->n_dims > 1) {
          gating_projection_dim = gating_input->dims[1];
        }
        const int64_t gating_dim = gating_projection_dim / 2;
        supported =
            supported &&
            detail::tensor_shape(output_projection, dep_dim, dep_dim) &&
            detail::supported_mul_mat_dtype(
                static_cast<detail::dtype>(output_projection->type)) &&
            gating_projection_dim > 0 && (gating_projection_dim % 2) == 0 &&
            static_cast<uint64_t>(gating_projection_dim) <=
                ctx.capacity.hidden_dim * 8u &&
            detail::tensor_shape(gating_input, dep_dim,
                                 gating_projection_dim) &&
            detail::supported_mul_mat_dtype(
                static_cast<detail::dtype>(gating_input->type)) &&
            gating_dim > 0 &&
            static_cast<uint64_t>(gating_dim) <= ctx.capacity.hidden_dim * 4u &&
            detail::tensor_shape(gating_output, gating_dim, dep_dim) &&
            detail::supported_mul_mat_dtype(
                static_cast<detail::dtype>(gating_output->type));
      }
    }
    return supported;
  }
};

struct guard_bound_root_operands_unsupported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_bound_root_operands_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_split_projection_layout_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    bool supported = lm.num_layers > 0;
    for (int32_t layer_index = 0; layer_index < lm.num_layers; ++layer_index) {
      const auto *projection =
          ctx.session.contract.lm
              .temporal_layers[static_cast<size_t>(layer_index)]
              .split_input_projection.tensor;
      supported = supported &&
                  detail::tensor_shape(projection, lm.dim, lm.dim * 3) &&
                  detail::supported_mul_mat_dtype(
                      static_cast<detail::dtype>(projection->type));
    }
    return supported;
  }
};

struct guard_temporal_fused_projection_layout_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    if (guard_temporal_split_projection_layout_supported{}(runtime_ev, ctx)) {
      return false;
    }
    const auto &lm = runtime_ev.request.model.moshi_lm;
    bool supported = lm.num_layers > 0;
    for (int32_t layer_index = 0; layer_index < lm.num_layers; ++layer_index) {
      const auto *projection =
          ctx.session.contract.lm
              .temporal_layers[static_cast<size_t>(layer_index)]
              .fused_input_projection.tensor;
      supported = supported &&
                  detail::tensor_shape(projection, lm.dim, lm.dim * 3) &&
                  detail::supported_mul_mat_dtype(
                      static_cast<detail::dtype>(projection->type));
    }
    return supported;
  }
};

struct guard_temporal_projection_layout_unsupported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_split_projection_layout_supported{}(runtime_ev,
                                                               ctx) &&
           !guard_temporal_fused_projection_layout_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_split_projection_layout_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    bool supported = lm.depformer_num_layers > 0 && lm.dep_q > 0;
    for (int32_t layer_index = 0; layer_index < lm.depformer_num_layers;
         ++layer_index) {
      for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
        const auto *projection =
            ctx.session.contract.lm
                .depformer_layers[static_cast<size_t>(layer_index)]
                .codebooks[static_cast<size_t>(codebook)]
                .split_input_projection.tensor;
        supported = supported &&
                    detail::tensor_shape(projection, lm.depformer_dim,
                                         lm.depformer_dim * 3) &&
                    detail::supported_mul_mat_dtype(
                        static_cast<detail::dtype>(projection->type));
      }
    }
    return supported;
  }
};

struct guard_depformer_fused_projection_layout_supported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    if (guard_depformer_split_projection_layout_supported{}(runtime_ev, ctx)) {
      return false;
    }
    const auto &lm = runtime_ev.request.model.moshi_lm;
    bool supported = lm.depformer_num_layers > 0 && lm.dep_q > 0;
    for (int32_t layer_index = 0; layer_index < lm.depformer_num_layers;
         ++layer_index) {
      for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
        const auto *projection =
            ctx.session.contract.lm
                .depformer_layers[static_cast<size_t>(layer_index)]
                .codebooks[static_cast<size_t>(codebook)]
                .fused_input_projection.tensor;
        supported = supported &&
                    detail::tensor_shape(projection, lm.depformer_dim,
                                         lm.depformer_dim * 3) &&
                    detail::supported_mul_mat_dtype(
                        static_cast<detail::dtype>(projection->type));
      }
    }
    return supported;
  }
};

struct guard_depformer_projection_layout_unsupported {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_split_projection_layout_supported{}(runtime_ev,
                                                                ctx) &&
           !guard_depformer_fused_projection_layout_supported{}(runtime_ev,
                                                                ctx);
  }
};

struct guard_sampling_seed_nonzero {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sampling_seed != 0u;
  }
};

struct guard_sampling_seed_zero {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sampling_seed == 0u;
  }
};

struct guard_text_sampling_top_k_within_card {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sampling_text_top_k <=
           runtime_ev.request.model.moshi_lm.text_card;
  }
};

struct guard_text_sampling_top_k_exceeds_card {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sampling_text_top_k >
           runtime_ev.request.model.moshi_lm.text_card;
  }
};

struct guard_audio_sampling_top_k_within_card {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sampling_audio_top_k <=
           runtime_ev.request.model.moshi_lm.card;
  }
};

struct guard_audio_sampling_top_k_exceeds_card {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sampling_audio_top_k >
           runtime_ev.request.model.moshi_lm.card;
  }
};

struct guard_step_model_matches {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return ctx.session.model == &runtime_ev.request.model;
  }
};

struct guard_step_model_mismatch {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_step_model_matches{}(runtime_ev, ctx);
  }
};

struct guard_step_shape_valid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const bool full = runtime_ev.request.phase ==
                      emel::speech::predictor::moshi::event::graph_phase::full;
    const bool prediction =
        runtime_ev.request.phase ==
        emel::speech::predictor::moshi::event::graph_phase::prediction;
    return (full || prediction) &&
           runtime_ev.request.input_sequence.data() != nullptr &&
           runtime_ev.request.input_sequence.size() ==
               static_cast<size_t>(ctx.session.codebook_count) &&
           runtime_ev.request.audio_tokens_out.data() != nullptr &&
           runtime_ev.request.audio_tokens_out.size() >=
               static_cast<size_t>(ctx.session.dep_q) &&
           (full || runtime_ev.request.temporal_state.size() >=
                        static_cast<size_t>(ctx.session.hidden_dim));
  }
};

struct guard_sampling_step_valid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.request.phase ==
               emel::speech::predictor::moshi::event::graph_phase::sampling &&
           guard_step_model_matches{}(runtime_ev, ctx) &&
           runtime_ev.request.input_sequence.data() != nullptr &&
           runtime_ev.request.input_sequence.size() ==
               static_cast<size_t>(ctx.session.codebook_count) &&
           runtime_ev.request.audio_tokens_out.data() != nullptr &&
           runtime_ev.request.audio_tokens_out.size() >=
               static_cast<size_t>(ctx.session.dep_q) &&
           runtime_ev.request.temporal_state.size() >=
               static_cast<size_t>(ctx.session.hidden_dim);
  }
};

struct guard_sampling_step_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_sampling_step_valid{}(runtime_ev, ctx);
  }
};

struct guard_full_graph_phase {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.phase ==
           emel::speech::predictor::moshi::event::graph_phase::full;
  }
};

struct guard_prediction_graph_phase {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.phase ==
           emel::speech::predictor::moshi::event::graph_phase::prediction;
  }
};

struct guard_step_shape_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_step_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_external_input_embedding_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.request.input_embedding.data() != nullptr &&
           runtime_ev.request.input_embedding.size() ==
               static_cast<size_t>(ctx.session.hidden_dim) &&
           ctx.session.hidden_dim > 0 &&
           static_cast<uint64_t>(ctx.session.hidden_dim) <=
               ctx.capacity.hidden_dim;
  }
};

struct guard_token_input_embedding_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    bool tokens_ok =
        runtime_ev.request.input_sequence[0] != ctx.policy.token_zero &&
        detail::token_in_embedding_range(runtime_ev.request.input_sequence[0],
                                         lm.text_card, ctx.policy.token_zero);
    for (int32_t index = 0; index < lm.n_q; ++index) {
      tokens_ok =
          tokens_ok &&
          detail::token_in_embedding_range(
              runtime_ev.request.input_sequence[static_cast<size_t>(index + 1)],
              lm.card, ctx.policy.token_zero);
    }
    return runtime_ev.request.input_embedding.data() == nullptr &&
           !lm.demux_second_stream && hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= ctx.capacity.hidden_dim &&
           tokens_ok;
  }
};

struct guard_input_embedding_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_external_input_embedding_supported{}(runtime_ev, ctx) ||
           guard_token_input_embedding_supported{}(runtime_ev, ctx);
  }
};

struct guard_input_embedding_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_input_embedding_supported{}(runtime_ev, ctx);
  }
};

struct guard_token_input_embedding_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_external_input_embedding_supported{}(runtime_ev, ctx) &&
           !guard_token_input_embedding_supported{}(runtime_ev, ctx);
  }
};

struct guard_input_text_embedding_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.input_text_embedding_ok;
  }
};

struct guard_input_text_embedding_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_input_text_embedding_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_embedding_view_bound {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.embedding_view_bound;
  }
};

struct guard_embedding_view_bind_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_embedding_view_bound{}(runtime_ev, ctx);
  }
};

struct guard_embedding_row_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.embedding_row_ok;
  }
};

struct guard_embedding_row_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_embedding_row_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_projection_view_bound {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.projection_view_bound;
  }
};

struct guard_projection_view_bind_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_projection_view_bound{}(runtime_ev, ctx);
  }
};

struct guard_current_input_audio_token_present {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t index = runtime_ev.ctx.input_audio_codebook_index;
    if (index < 0 || index >= lm.n_q) {
      return false;
    }
    const int32_t token =
        runtime_ev.request.input_sequence[static_cast<size_t>(index + 1)];
    return token != ctx.policy.token_zero &&
           detail::token_in_embedding_range(token, lm.card,
                                            ctx.policy.token_zero) &&
           guard_input_embedding_supported{}(runtime_ev, ctx);
  }
};

struct guard_current_input_audio_token_zero {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t index = runtime_ev.ctx.input_audio_codebook_index;
    if (index < 0 || index >= lm.n_q) {
      return false;
    }
    return runtime_ev.request.input_sequence[static_cast<size_t>(index + 1)] ==
               ctx.policy.token_zero &&
           guard_input_embedding_supported{}(runtime_ev, ctx);
  }
};

struct guard_current_input_audio_token_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_current_input_audio_token_present{}(runtime_ev, ctx) &&
           !guard_current_input_audio_token_zero{}(runtime_ev, ctx);
  }
};

struct guard_input_audio_embedding_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.input_audio_embedding_ok;
  }
};

struct guard_input_audio_embedding_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_input_audio_embedding_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_more_input_audio_codebooks {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.input_audio_embedding_ok &&
           runtime_ev.ctx.input_audio_codebook_index + 1 <
               runtime_ev.request.model.moshi_lm.n_q;
  }
};

struct guard_input_audio_codebooks_complete {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.ctx.input_audio_embedding_ok &&
           !guard_more_input_audio_codebooks{}(runtime_ev, ctx);
  }
};

struct guard_input_embedding_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.input_embedding_ok;
  }
};

struct guard_input_embedding_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_input_embedding_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_kv_binding_present {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    (void)runtime_ev;
    return !ctx.temporal_kv.key_cache.empty() &&
           !ctx.temporal_kv.value_cache.empty() &&
           !ctx.temporal_kv.layer_cache_offsets.empty() &&
           ctx.temporal_positions != nullptr;
  }
};

struct guard_temporal_kv_binding_missing {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_kv_binding_present{}(runtime_ev, ctx);
  }
};

struct guard_temporal_kv_bound {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &view = runtime_ev.ctx.temporal_kv;
    return runtime_ev.ctx.temporal_kv_bound && !view.key_cache.empty() &&
           !view.value_cache.empty() && !view.layer_cache_offsets.empty() &&
           view.layer_count > 0 && view.position_capacity > 0 &&
           view.kv_dim > 0;
  }
};

struct guard_temporal_position_advance_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &result = runtime_ev.ctx.temporal_position;
    const auto &window = result.window;
    return runtime_ev.ctx.temporal_position_accepted &&
           runtime_ev.ctx.temporal_position_error ==
               static_cast<int32_t>(
                   emel::error::cast(emel::memory::streaming::error::none)) &&
           result.logical_position >= 0 && result.physical_position >= 0 &&
           result.physical_position < window.capacity && window.capacity > 0 &&
           window.logical_end == result.logical_position + 1 &&
           window.valid_positions > 0 &&
           window.valid_positions <= window.capacity;
  }
};

struct guard_temporal_position_advance_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_position_advance_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_kv_bind_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_kv_bound{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_norm_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    return layer >= 0 && layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= ctx.capacity.hidden_dim;
  }
};

struct guard_temporal_layer_norm_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_norm_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_norm_rms_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_norm_rms_ok;
  }
};

struct guard_temporal_layer_norm_rms_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_norm_rms_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_norm_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_norm_ok;
  }
};

struct guard_temporal_layer_norm_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_norm_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_projection_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    return layer >= 0 && layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= ctx.capacity.hidden_dim &&
           static_cast<uint64_t>(hidden_dim) * 3u <=
               ctx.capacity.hidden_dim * 3u;
  }
};

struct guard_temporal_layer_projection_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_projection_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_projection_ok;
  }
};

struct guard_temporal_layer_projection_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_projection_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_rope_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const auto &position = runtime_ev.ctx.temporal_position;
    const int32_t head_dim = lm.num_heads > 0 ? hidden_dim / lm.num_heads : 0;
    return runtime_ev.ctx.temporal_layer_projection_ok && hidden_dim > 0 &&
           lm.num_heads > 0 && lm.max_period > 0 &&
           hidden_dim % lm.num_heads == 0 && head_dim > 0 &&
           (head_dim % 2) == 0 &&
           static_cast<uint64_t>(head_dim) <= ctx.capacity.hidden_dim &&
           position.logical_position >= 0 &&
           position.logical_position <= std::numeric_limits<int32_t>::max() &&
           position.physical_position >= 0 &&
           position.physical_position <
               runtime_ev.ctx.temporal_kv.position_capacity;
  }
};

struct guard_temporal_layer_rope_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_rope_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_rope_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_rope_ok;
  }
};

struct guard_temporal_layer_rope_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_rope_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_cache_write_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &view = runtime_ev.ctx.temporal_kv;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    if (!runtime_ev.ctx.temporal_layer_rope_ok || hidden_dim <= 0 ||
        view.kv_dim != hidden_dim || layer < 0 || layer >= view.layer_count ||
        view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size() ||
        runtime_ev.ctx.temporal_position.physical_position < 0) {
      return false;
    }
    const size_t physical_position =
        static_cast<size_t>(runtime_ev.ctx.temporal_position.physical_position);
    const size_t dim = static_cast<size_t>(hidden_dim);
    return physical_position < static_cast<size_t>(view.position_capacity) &&
           guard_cache_span_valid(
               view.layer_cache_offsets[static_cast<size_t>(layer)],
               physical_position, 1u, dim, view.key_cache.size(),
               view.value_cache.size());
  }
};

struct guard_temporal_layer_cache_write_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_cache_write_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_cache_write_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_cache_write_ok;
  }
};

struct guard_temporal_layer_cache_write_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_cache_write_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_attention_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &view = runtime_ev.ctx.temporal_kv;
    const auto &window = runtime_ev.ctx.temporal_position.window;
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t capacity = view.position_capacity;
    const int32_t head_dim = lm.num_heads > 0 ? hidden_dim / lm.num_heads : 0;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    if (!runtime_ev.ctx.temporal_layer_cache_write_ok || hidden_dim <= 0 ||
        lm.num_heads <= 0 || hidden_dim % lm.num_heads != 0 || head_dim <= 0 ||
        static_cast<uint64_t>(head_dim) > ctx.capacity.hidden_dim ||
        window.valid_positions <= 0 || capacity <= 0 ||
        static_cast<uint64_t>(capacity) > ctx.capacity.temporal_context ||
        window.capacity != capacity || window.valid_positions > capacity ||
        window.logical_begin < 0 ||
        window.logical_end - window.logical_begin != window.valid_positions ||
        window.physical_begin < 0 || window.physical_begin >= capacity ||
        view.kv_dim != hidden_dim || layer < 0 || layer >= view.layer_count ||
        view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size()) {
      return false;
    }
    const size_t layer_offset =
        view.layer_cache_offsets[static_cast<size_t>(layer)];
    const size_t dim = static_cast<size_t>(hidden_dim);
    return guard_cache_span_valid(
        layer_offset, 0u, static_cast<size_t>(capacity), dim,
        view.key_cache.size(), view.value_cache.size());
  }
};

struct guard_temporal_layer_attention_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_attention_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_attention_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_attention_ok;
  }
};

struct guard_temporal_layer_attention_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_attention_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_out_projection_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    return runtime_ev.ctx.temporal_layer_attention_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0;
  }
};

struct guard_temporal_layer_out_projection_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_out_projection_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_out_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_out_projection_ok;
  }
};

struct guard_temporal_layer_out_projection_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_out_projection_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_residual_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_residual_ok;
  }
};

struct guard_temporal_layer_residual_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_residual_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_norm2_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    return runtime_ev.ctx.temporal_layer_residual_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= ctx.capacity.hidden_dim;
  }
};

struct guard_temporal_layer_norm2_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_norm2_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_norm2_rms_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_norm2_rms_ok;
  }
};

struct guard_temporal_layer_norm2_rms_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_norm2_rms_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_norm2_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_norm2_ok;
  }
};

struct guard_temporal_layer_norm2_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_norm2_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_gating_in_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    return runtime_ev.ctx.temporal_layer_norm2_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0;
  }
};

struct guard_temporal_layer_gating_in_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_gating_in_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_gating_in_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_gating_in_ok;
  }
};

struct guard_temporal_layer_gating_in_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_gating_in_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_silu_gate_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_gating_in_ok &&
           runtime_ev.ctx.temporal_layer_index >= 0 &&
           runtime_ev.ctx.temporal_layer_index <
               runtime_ev.request.model.moshi_lm.num_layers;
  }
};

struct guard_temporal_layer_silu_gate_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_silu_gate_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_silu_gate_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_silu_gate_ok;
  }
};

struct guard_temporal_layer_silu_gate_silu_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_silu_gate_silu_ok;
  }
};

struct guard_temporal_layer_silu_gate_silu_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_silu_gate_silu_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_silu_gate_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_silu_gate_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_gating_out_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    return runtime_ev.ctx.temporal_layer_silu_gate_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0;
  }
};

struct guard_temporal_layer_gating_out_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_gating_out_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_gating_out_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_gating_out_ok;
  }
};

struct guard_temporal_layer_gating_out_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_gating_out_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_layer_ff_residual_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_ff_residual_ok;
  }
};

struct guard_temporal_layer_ff_residual_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_layer_ff_residual_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_more_temporal_layers {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_layer_ff_residual_ok &&
           runtime_ev.ctx.temporal_layer_index + 1 <
               runtime_ev.request.model.moshi_lm.num_layers;
  }
};

struct guard_temporal_layers_complete {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.ctx.temporal_layer_ff_residual_ok &&
           !guard_more_temporal_layers{}(runtime_ev, ctx);
  }
};

struct guard_temporal_out_norm_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    return guard_temporal_layers_complete{}(runtime_ev, ctx) &&
           hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= ctx.capacity.hidden_dim;
  }
};

struct guard_temporal_out_norm_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_out_norm_supported{}(runtime_ev, ctx);
  }
};

struct guard_temporal_out_norm_rms_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_out_norm_rms_ok;
  }
};

struct guard_temporal_out_norm_rms_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_out_norm_rms_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_temporal_out_norm_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_out_norm_ok;
  }
};

struct guard_temporal_out_norm_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_temporal_out_norm_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_text_logits_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    return runtime_ev.ctx.temporal_out_norm_ok && hidden_dim > 0 &&
           lm.text_card > 0;
  }
};

struct guard_text_logits_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_text_logits_supported{}(runtime_ev, ctx);
  }
};

struct guard_forced_text_token_valid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.forced_text_token >= 0 &&
           runtime_ev.request.forced_text_token <
               runtime_ev.request.model.moshi_lm.text_card;
  }
};

struct guard_forced_text_token_absent {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_forced_text_token_valid{}(runtime_ev, ctx);
  }
};

struct guard_text_sampling_config_valid {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return ctx.sampling.enabled && ctx.sampler != nullptr &&
           ctx.sampling.text_temperature > 0.0f &&
           ctx.sampling.text_top_k > 0 &&
           ctx.sampling.text_top_k <= ctx.session.text_card &&
           static_cast<uint64_t>(ctx.session.text_card) <=
               ctx.capacity.sampling_card &&
           static_cast<uint64_t>(ctx.sampling.text_top_k) <=
               ctx.capacity.sampling_top_k;
  }
};

struct guard_text_sampling_config_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return ctx.sampling.enabled && ctx.sampling.text_temperature > 0.0f &&
           !guard_text_sampling_config_valid{}(runtime_ev, ctx);
  }
};

struct guard_forced_text_token_valid_and_sampling_consume {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_forced_text_token_valid{}(runtime_ev, ctx) &&
           ctx.sampling.consume_forced_text &&
           guard_text_sampling_config_valid{}(runtime_ev, ctx);
  }
};

struct guard_forced_text_token_valid_without_sampling_consume {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_forced_text_token_valid{}(runtime_ev, ctx) &&
           (!ctx.sampling.consume_forced_text || !ctx.sampling.enabled ||
            ctx.sampling.text_temperature <= 0.0f);
  }
};

struct guard_forced_text_sampling_config_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_forced_text_token_valid{}(runtime_ev, ctx) &&
           ctx.sampling.consume_forced_text &&
           guard_text_sampling_config_invalid{}(runtime_ev, ctx);
  }
};

struct guard_text_logits_projection_bound_and_no_forced_token {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_projection_view_bound{}(runtime_ev, ctx) &&
           guard_forced_text_token_absent{}(runtime_ev, ctx);
  }
};

struct guard_text_logits_projection_bound_and_no_forced_token_argmax {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_text_logits_projection_bound_and_no_forced_token{}(runtime_ev,
                                                                    ctx) &&
           (!ctx.sampling.enabled || ctx.sampling.text_temperature <= 0.0f);
  }
};

struct guard_text_logits_projection_bound_and_no_forced_token_sampling {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_text_logits_projection_bound_and_no_forced_token{}(runtime_ev,
                                                                    ctx) &&
           guard_text_sampling_config_valid{}(runtime_ev, ctx);
  }
};

struct guard_text_logits_projection_bound_and_no_forced_token_sampling_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_text_logits_projection_bound_and_no_forced_token{}(runtime_ev,
                                                                    ctx) &&
           guard_text_sampling_config_invalid{}(runtime_ev, ctx);
  }
};

struct guard_text_logits_matmul_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.text_logits_ok;
  }
};

struct guard_text_logits_matmul_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_text_logits_matmul_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_text_logits_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.text_logits_ok && runtime_ev.ctx.best_index >= 0;
  }
};

struct guard_text_logits_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_text_logits_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_text_sampling_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.sampler_accepted &&
           runtime_ev.ctx.sampler_error ==
               emel::error::cast(emel::logits::sampler::error::none);
  }
};

struct guard_text_sampling_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_text_sampling_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_sampled_text_token_ready {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_text_sampling_succeeded{}(runtime_ev, ctx) &&
           guard_forced_text_token_absent{}(runtime_ev, ctx);
  }
};

struct guard_forced_text_sampling_consumed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_text_sampling_succeeded{}(runtime_ev, ctx) &&
           guard_forced_text_token_valid_and_sampling_consume{}(runtime_ev,
                                                                ctx);
  }
};

struct guard_depformer_kv_binding_present {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    (void)runtime_ev;
    return !ctx.depformer_kv.key_cache.empty() &&
           !ctx.depformer_kv.value_cache.empty() &&
           !ctx.depformer_kv.layer_cache_offsets.empty() &&
           ctx.depformer_positions != nullptr;
  }
};

struct guard_depformer_kv_binding_missing {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_kv_binding_present{}(runtime_ev, ctx);
  }
};

struct guard_depformer_kv_bound {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &view = runtime_ev.ctx.depformer_kv;
    return runtime_ev.ctx.depformer_kv_bound && !view.key_cache.empty() &&
           !view.value_cache.empty() && !view.layer_cache_offsets.empty() &&
           view.layer_count > 0 && view.position_capacity > 0 &&
           view.kv_dim > 0;
  }
};

struct guard_depformer_position_reset_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_position_accepted &&
           runtime_ev.ctx.depformer_position_error ==
               static_cast<int32_t>(
                   emel::error::cast(emel::memory::streaming::error::none));
  }
};

struct guard_depformer_position_reset_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_position_reset_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_position_advance_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &result = runtime_ev.ctx.depformer_position;
    const auto &window = result.window;
    return runtime_ev.ctx.depformer_position_accepted &&
           runtime_ev.ctx.depformer_position_error ==
               static_cast<int32_t>(
                   emel::error::cast(emel::memory::streaming::error::none)) &&
           result.logical_position >= 0 && result.physical_position >= 0 &&
           result.physical_position < window.capacity && window.capacity > 0 &&
           window.logical_end == result.logical_position + 1 &&
           window.valid_positions > 0 &&
           window.valid_positions <= window.capacity;
  }
};

struct guard_depformer_position_advance_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_position_advance_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_kv_bind_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_kv_bound{}(runtime_ev, ctx);
  }
};

struct guard_depformer_scheduled_weight_present {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    if (lm.depformer_weight_schedule_count == 0u || codebook < 0 ||
        static_cast<uint32_t>(codebook) >= lm.depformer_weight_schedule_count) {
      return false;
    }
    const int32_t weight_index =
        lm.depformer_weight_schedule[static_cast<size_t>(codebook)];
    return weight_index >= 0 && weight_index < lm.dep_q;
  }
};

struct guard_depformer_scheduled_weight_absent {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.model.moshi_lm.depformer_weight_schedule_count ==
           0u;
  }
};

struct guard_depformer_scheduled_weight_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_scheduled_weight_absent{}(runtime_ev, ctx) &&
           !guard_depformer_scheduled_weight_present{}(runtime_ev, ctx);
  }
};

struct guard_depformer_text_input_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    if (codebook != 0) {
      return false;
    }
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_kv_bound && runtime_ev.ctx.text_logits_ok &&
           codebook == 0 && hidden_dim > 0 && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= ctx.capacity.hidden_dim &&
           detail::token_in_embedding_range(runtime_ev.request.text_token_out,
                                            lm.text_card,
                                            ctx.policy.token_zero);
  }
};

struct guard_depformer_audio_input_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    if (codebook <= 0 || codebook >= lm.dep_q) {
      return false;
    }
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_kv_bound && runtime_ev.ctx.text_logits_ok &&
           codebook > 0 && codebook < lm.dep_q && hidden_dim > 0 &&
           dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= ctx.capacity.hidden_dim &&
           detail::token_in_embedding_range(
               runtime_ev.request
                   .audio_tokens_out[static_cast<size_t>(codebook - 1)],
               lm.card, ctx.policy.token_zero);
  }
};

struct guard_depformer_input_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_text_input_supported{}(runtime_ev, ctx) &&
           !guard_depformer_audio_input_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_input_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_input_ok;
  }
};

struct guard_depformer_input_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_input_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_input_projection_bound {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_input_projection_bound;
  }
};

struct guard_depformer_input_projection_bind_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_input_projection_bound{}(runtime_ev, ctx);
  }
};

struct guard_depformer_input_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_input_projection_ok;
  }
};

struct guard_depformer_input_projection_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_input_projection_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_text_input_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_depformer_input_projection_succeeded{}(runtime_ev, ctx) &&
           guard_depformer_text_input_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_audio_input_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_depformer_input_projection_succeeded{}(runtime_ev, ctx) &&
           guard_depformer_audio_input_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_input_projection_embedding_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_depformer_input_projection_succeeded{}(runtime_ev, ctx) &&
           !guard_depformer_text_input_supported{}(runtime_ev, ctx) &&
           !guard_depformer_audio_input_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_norm_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    return runtime_ev.ctx.depformer_input_ok && layer >= 0 &&
           layer < lm.depformer_num_layers && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= ctx.capacity.hidden_dim;
  }
};

struct guard_depformer_layer_norm_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_norm_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_norm_rms_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_norm_rms_ok;
  }
};

struct guard_depformer_layer_norm_rms_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_norm_rms_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_norm_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_norm_ok;
  }
};

struct guard_depformer_layer_norm_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_norm_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_projection_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_layer_norm_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) * 3u <= ctx.capacity.hidden_dim * 3u;
  }
};

struct guard_depformer_layer_projection_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_projection_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_projection_ok;
  }
};

struct guard_depformer_layer_projection_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_projection_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_cache_write_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &view = runtime_ev.ctx.depformer_kv;
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t physical =
        runtime_ev.ctx.depformer_position.physical_position;
    if (!runtime_ev.ctx.depformer_layer_projection_ok || dep_dim <= 0 ||
        view.kv_dim != dep_dim || layer < 0 || layer >= view.layer_count ||
        view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size() ||
        view.position_capacity <= 0 || physical < 0 ||
        physical >= view.position_capacity) {
      return false;
    }
    return guard_cache_span_valid(
        view.layer_cache_offsets[static_cast<size_t>(layer)],
        static_cast<size_t>(physical), 1u, static_cast<size_t>(dep_dim),
        view.key_cache.size(), view.value_cache.size());
  }
};

struct guard_depformer_layer_cache_write_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_cache_write_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_cache_write_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_cache_write_ok;
  }
};

struct guard_depformer_layer_cache_write_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_cache_write_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_attention_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const auto &view = runtime_ev.ctx.depformer_kv;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t head_dim =
        lm.depformer_num_heads > 0 ? dep_dim / lm.depformer_num_heads : 0;
    const auto &window = runtime_ev.ctx.depformer_position.window;
    const int32_t valid_positions = window.valid_positions;
    const int32_t capacity = view.position_capacity;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    if (!runtime_ev.ctx.depformer_layer_cache_write_ok || dep_dim <= 0 ||
        lm.depformer_num_heads <= 0 || dep_dim % lm.depformer_num_heads != 0 ||
        head_dim <= 0 ||
        static_cast<uint64_t>(head_dim) > ctx.capacity.hidden_dim ||
        valid_positions <= 0 || capacity <= 0 ||
        static_cast<uint64_t>(capacity) > ctx.capacity.depformer_context ||
        valid_positions > capacity || window.capacity != capacity ||
        window.logical_begin < 0 ||
        window.logical_end - window.logical_begin != valid_positions ||
        window.physical_begin < 0 || window.physical_begin >= capacity ||
        view.kv_dim != dep_dim || layer < 0 || layer >= view.layer_count ||
        view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size()) {
      return false;
    }
    const size_t dim = static_cast<size_t>(dep_dim);
    return guard_cache_span_valid(
        view.layer_cache_offsets[static_cast<size_t>(layer)], 0u,
        static_cast<size_t>(capacity), dim, view.key_cache.size(),
        view.value_cache.size());
  }
};

struct guard_depformer_layer_attention_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_attention_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_attention_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_attention_ok;
  }
};

struct guard_depformer_layer_attention_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_attention_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_out_projection_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_layer_attention_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0;
  }
};

struct guard_depformer_layer_out_projection_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_out_projection_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_out_projection_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_out_projection_ok;
  }
};

struct guard_depformer_layer_out_projection_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_out_projection_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_residual_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_residual_ok;
  }
};

struct guard_depformer_layer_residual_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_residual_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_norm2_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    return runtime_ev.ctx.depformer_layer_residual_ok && layer >= 0 &&
           layer < lm.depformer_num_layers && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= ctx.capacity.hidden_dim;
  }
};

struct guard_depformer_layer_norm2_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_norm2_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_norm2_rms_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_norm2_rms_ok;
  }
};

struct guard_depformer_layer_norm2_rms_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_norm2_rms_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_norm2_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_norm2_ok;
  }
};

struct guard_depformer_layer_norm2_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_norm2_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_gating_in_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_layer_norm2_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0;
  }
};

struct guard_depformer_layer_gating_in_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_gating_in_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_gating_in_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_gating_in_ok;
  }
};

struct guard_depformer_layer_gating_in_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_gating_in_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_silu_gate_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q || codebook < 0 ||
        codebook >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_layer_gating_in_ok &&
           runtime_ev.ctx.depformer_layer_index >= 0 &&
           runtime_ev.ctx.depformer_layer_index <
               runtime_ev.request.model.moshi_lm.depformer_num_layers;
  }
};

struct guard_depformer_layer_silu_gate_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_silu_gate_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_silu_gate_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_silu_gate_ok;
  }
};

struct guard_depformer_layer_silu_gate_silu_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_silu_gate_silu_ok;
  }
};

struct guard_depformer_layer_silu_gate_silu_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_silu_gate_silu_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_silu_gate_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_silu_gate_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_gating_out_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    return runtime_ev.ctx.depformer_layer_silu_gate_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0;
  }
};

struct guard_depformer_layer_gating_out_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_gating_out_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_gating_out_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_gating_out_ok;
  }
};

struct guard_depformer_layer_gating_out_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_gating_out_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_layer_ff_residual_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_ff_residual_ok;
  }
};

struct guard_depformer_layer_ff_residual_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_layer_ff_residual_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_more_depformer_layers {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_layer_ff_residual_ok &&
           runtime_ev.ctx.depformer_layer_index + 1 <
               runtime_ev.request.model.moshi_lm.depformer_num_layers;
  }
};

struct guard_depformer_layers_complete {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.ctx.depformer_layer_ff_residual_ok &&
           !guard_more_depformer_layers{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    return guard_depformer_layers_complete{}(runtime_ev, ctx) &&
           codebook >= 0 && codebook < lm.dep_q && dep_dim > 0 && lm.card > 0;
  }
};

struct guard_depformer_logits_unsupported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_logits_supported{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_logits_ok &&
           runtime_ev.ctx.best_index >= 0 &&
           runtime_ev.ctx.best_index < runtime_ev.request.model.moshi_lm.card;
  }
};

struct guard_depformer_logits_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_logits_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_sampling_config_valid {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return ctx.sampling.enabled && ctx.sampler != nullptr &&
           ctx.sampling.audio_temperature > 0.0f &&
           ctx.sampling.audio_top_k > 0 &&
           ctx.sampling.audio_top_k <= ctx.session.audio_card &&
           static_cast<uint64_t>(ctx.session.audio_card) <=
               ctx.capacity.sampling_card &&
           static_cast<uint64_t>(ctx.sampling.audio_top_k) <=
               ctx.capacity.sampling_top_k;
  }
};

struct guard_depformer_sampling_config_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return ctx.sampling.enabled && ctx.sampling.audio_temperature > 0.0f &&
           !guard_depformer_sampling_config_valid{}(runtime_ev, ctx);
  }
};

struct guard_depformer_sampling_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.sampler_accepted &&
           runtime_ev.ctx.sampler_error ==
               emel::error::cast(emel::logits::sampler::error::none);
  }
};

struct guard_depformer_sampling_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_sampling_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_argmax {
  bool operator()(const event::step_run &,
                  const action::context &ctx) const noexcept {
    return !ctx.sampling.enabled || ctx.sampling.audio_temperature <= 0.0f;
  }
};

struct guard_depformer_logits_sampling {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_depformer_sampling_config_valid{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_sampling_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_depformer_sampling_config_invalid{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_projection_bound_argmax {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_projection_view_bound{}(runtime_ev, ctx) &&
           guard_depformer_logits_argmax{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_projection_bound_sampling {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_projection_view_bound{}(runtime_ev, ctx) &&
           guard_depformer_logits_sampling{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_projection_bound_sampling_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_projection_view_bound{}(runtime_ev, ctx) &&
           guard_depformer_logits_sampling_invalid{}(runtime_ev, ctx);
  }
};

struct guard_depformer_logits_matmul_succeeded {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_logits_ok;
  }
};

struct guard_depformer_logits_matmul_failed {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_depformer_logits_matmul_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_reset_temporal_positions_present {
  bool operator()(const event::reset_run &,
                  const action::context &ctx) const noexcept {
    return ctx.temporal_positions != nullptr;
  }
};

struct guard_reset_temporal_positions_missing {
  bool operator()(const event::reset_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_reset_temporal_positions_present{}(runtime_ev, ctx);
  }
};

struct guard_reset_temporal_positions_succeeded {
  bool operator()(const event::reset_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.temporal_position_accepted &&
           runtime_ev.ctx.temporal_position_error ==
               static_cast<int32_t>(emel::error::cast(
                   emel::memory::streaming::error::none));
  }
};

struct guard_reset_temporal_positions_failed {
  bool operator()(const event::reset_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_reset_temporal_positions_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_reset_depformer_positions_present {
  bool operator()(const event::reset_run &,
                  const action::context &ctx) const noexcept {
    return ctx.depformer_positions != nullptr;
  }
};

struct guard_reset_depformer_positions_missing {
  bool operator()(const event::reset_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_reset_depformer_positions_present{}(runtime_ev, ctx);
  }
};

struct guard_reset_depformer_positions_succeeded {
  bool operator()(const event::reset_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_position_accepted &&
           runtime_ev.ctx.depformer_position_error ==
               static_cast<int32_t>(emel::error::cast(
                   emel::memory::streaming::error::none));
  }
};

struct guard_reset_depformer_positions_failed {
  bool operator()(const event::reset_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_reset_depformer_positions_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_more_depformer_codebooks {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.depformer_logits_ok &&
           runtime_ev.ctx.depformer_codebook_index + 1 <
               runtime_ev.request.model.moshi_lm.dep_q;
  }
};

struct guard_depformer_codebooks_complete {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.ctx.depformer_logits_ok &&
           !guard_more_depformer_codebooks{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

template <class runtime_event_type> struct guard_no_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_out<runtime_event_type>{}(runtime_ev, ctx);
  }
};

struct guard_has_initialize_done_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_initialize_done_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_initialize_done_callback{}(runtime_ev, ctx);
  }
};

struct guard_has_initialize_error_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_initialize_error_callback {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_initialize_error_callback{}(runtime_ev, ctx);
  }
};

struct guard_unexpected_error_out_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    if constexpr (requires {
                    ev.ctx.err;
                    ev.request.error_out;
                  }) {
      return ev.request.error_out != nullptr;
    } else {
      return false;
    }
  }
};

struct guard_unexpected_error_out_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_unexpected_error_out_present{}(ev, ctx);
  }
};

} // namespace emel::speech::predictor::moshi::executor::guard
