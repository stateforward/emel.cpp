#pragma once

#include "emel/model/loader/errors.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/generator/moshi/executor/context.hpp"
#include "emel/speech/generator/moshi/executor/detail.hpp"
#include "emel/speech/generator/moshi/executor/events.hpp"

namespace emel::speech::generator::moshi::executor::guard {

struct guard_bind_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    return emel::model::moshi::detail::validate_execution_contract(model) ==
               emel::error::cast(emel::model::loader::error::none) &&
           model.moshi_component_id == emel::model::data::moshi_component::lm &&
           model.moshi_lm.n_q > 0 && model.moshi_lm.dep_q > 0 &&
           model.moshi_lm.dep_q <= model.moshi_lm.n_q &&
           model.moshi_lm.text_card > 0 && model.moshi_lm.card > 0 &&
           model.moshi_lm.dim > 0;
  }
};

struct guard_bind_contract_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_contract_valid{}(runtime_ev, ctx);
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
    return runtime_ev.request.input_sequence.data() != nullptr &&
           runtime_ev.request.input_sequence.size() ==
               static_cast<size_t>(ctx.session.codebook_count) &&
           runtime_ev.request.audio_tokens_out.data() != nullptr &&
           runtime_ev.request.audio_tokens_out.size() >=
               static_cast<size_t>(ctx.session.dep_q);
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
               detail::k_max_hidden_dim;
  }
};

struct guard_token_input_embedding_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const auto *text_emb = detail::find_tensor(model, "lm.text_emb.weight");
    const bool root_tensors_ok =
        detail::tensor_shape(text_emb, hidden_dim, lm.text_card + 1) &&
        detail::supported_get_rows_dtype(
            static_cast<detail::dtype>(text_emb->type));
    bool audio_tensors_ok = true;
    for (int32_t index = 0; index < lm.n_q; ++index) {
      const auto *audio_emb =
          detail::find_indexed_tensor(model, "lm.emb.%d.weight", index);
      audio_tensors_ok =
          audio_tensors_ok &&
          detail::tensor_shape(audio_emb, hidden_dim, lm.card + 1) &&
          detail::supported_get_rows_dtype(
              static_cast<detail::dtype>(audio_emb->type));
    }
    bool tokens_ok =
        runtime_ev.request.input_sequence[0] != detail::k_token_zero &&
        detail::token_in_embedding_range(runtime_ev.request.input_sequence[0],
                                         lm.text_card);
    for (int32_t index = 0; index < lm.n_q; ++index) {
      tokens_ok =
          tokens_ok &&
          detail::token_in_embedding_range(
              runtime_ev.request.input_sequence[static_cast<size_t>(index + 1)],
              lm.card);
    }
    return runtime_ev.request.input_embedding.data() == nullptr &&
           !lm.demux_second_stream && hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= detail::k_max_hidden_dim &&
           root_tensors_ok && audio_tensors_ok && tokens_ok;
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
    return token != detail::k_token_zero &&
           detail::token_in_embedding_range(token, lm.card) &&
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
               detail::k_token_zero &&
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
    return ctx.temporal_kv.cache != nullptr && ctx.temporal_kv.bind != nullptr;
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
           view.block_tokens > 0 && view.kv_dim > 0;
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
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *norm1 =
        detail::find_lm_transformer_tensor(model, layer, "norm1.alpha");
    return layer >= 0 && layer < model.moshi_lm.num_layers && hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(norm1, hidden_dim) &&
           static_cast<detail::dtype>(norm1->type) == detail::dtype::f32;
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
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *projection =
        detail::find_lm_transformer_projection(model, layer);
    return layer >= 0 && layer < model.moshi_lm.num_layers && hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= detail::k_max_hidden_dim &&
           static_cast<uint64_t>(hidden_dim) * 3u <= detail::k_max_qkv_dim &&
           detail::tensor_shape(projection, hidden_dim, hidden_dim * 3) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(projection->type));
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
    const int32_t logical_position = detail::current_logical_position(
        runtime_ev.request.memory_snapshot, runtime_ev.request.sequence_id);
    const int32_t physical_position = detail::physical_position(
        runtime_ev.request.memory_snapshot, runtime_ev.request.sequence_id,
        logical_position);
    const int32_t head_dim = lm.num_heads > 0 ? hidden_dim / lm.num_heads : 0;
    return runtime_ev.ctx.temporal_layer_projection_ok && hidden_dim > 0 &&
           lm.num_heads > 0 && lm.max_period > 0 &&
           hidden_dim % lm.num_heads == 0 && head_dim > 0 &&
           (head_dim % 2) == 0 &&
           static_cast<uint64_t>(head_dim) <= detail::k_max_hidden_dim &&
           logical_position >= 0 && physical_position >= 0 &&
           physical_position < runtime_ev.ctx.temporal_kv.position_capacity;
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
        runtime_ev.ctx.temporal_physical_position < 0) {
      return false;
    }
    const size_t layer_offset =
        view.layer_cache_offsets[static_cast<size_t>(layer)];
    const size_t physical_position =
        static_cast<size_t>(runtime_ev.ctx.temporal_physical_position);
    const size_t dim = static_cast<size_t>(hidden_dim);
    const size_t begin = layer_offset + physical_position * dim;
    const size_t end = begin + dim;
    return physical_position < static_cast<size_t>(view.position_capacity) &&
           end <= view.key_cache.size() && end <= view.value_cache.size();
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
    const auto &snapshot = runtime_ev.request.memory_snapshot;
    const auto &view = runtime_ev.ctx.temporal_kv;
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t sequence_length =
        snapshot.sequence_length(runtime_ev.request.sequence_id);
    const int32_t capacity = view.position_capacity;
    const int32_t head_dim = lm.num_heads > 0 ? hidden_dim / lm.num_heads : 0;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    if (!runtime_ev.ctx.temporal_layer_cache_write_ok || hidden_dim <= 0 ||
        lm.num_heads <= 0 || hidden_dim % lm.num_heads != 0 || head_dim <= 0 ||
        static_cast<uint64_t>(head_dim) > detail::k_max_hidden_dim ||
        sequence_length <= 0 || capacity <= 0 ||
        static_cast<uint64_t>(capacity) >
            detail::k_max_temporal_context ||
        view.kv_dim != hidden_dim || layer < 0 || layer >= view.layer_count ||
        view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size()) {
      return false;
    }
    const size_t layer_offset =
        view.layer_cache_offsets[static_cast<size_t>(layer)];
    const size_t dim = static_cast<size_t>(hidden_dim);
    const int32_t logical_begin =
        sequence_length > capacity ? sequence_length - capacity : 0;
    for (int32_t logical = logical_begin; logical < sequence_length; ++logical) {
      const int32_t physical = detail::physical_position(
          snapshot, runtime_ev.request.sequence_id, logical);
      if (physical < 0 ||
          physical >= capacity) {
        return false;
      }
      const size_t begin = layer_offset + static_cast<size_t>(physical) * dim;
      const size_t end = begin + dim;
      if (end > view.key_cache.size() || end > view.value_cache.size()) {
        return false;
      }
    }
    return true;
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
    const auto *projection = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, layer, "self_attn.out_projs.0.weight");
    return runtime_ev.ctx.temporal_layer_attention_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0 &&
           detail::tensor_shape(projection, hidden_dim, hidden_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(projection->type));
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
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *norm2 =
        detail::find_lm_transformer_tensor(model, layer, "norm2.alpha");
    return runtime_ev.ctx.temporal_layer_residual_ok && layer >= 0 &&
           layer < model.moshi_lm.num_layers && hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(norm2, hidden_dim) &&
           static_cast<detail::dtype>(norm2->type) == detail::dtype::f32;
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
    const auto *linear_in = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, layer, "gating.linear_in.weight");
    int64_t projection_dim = 0;
    if (linear_in != nullptr && linear_in->n_dims > 1) {
      projection_dim = linear_in->dims[1];
    }
    return runtime_ev.ctx.temporal_layer_norm2_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0 && projection_dim > 0 && (projection_dim % 2) == 0 &&
           static_cast<uint64_t>(projection_dim) <=
               detail::k_max_gating_projection_dim &&
           detail::tensor_shape(linear_in, hidden_dim, projection_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(linear_in->type));
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
    const auto *linear_in = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, runtime_ev.ctx.temporal_layer_index,
        "gating.linear_in.weight");
    int64_t projection_dim = 0;
    if (linear_in != nullptr && linear_in->n_dims > 1) {
      projection_dim = linear_in->dims[1];
    }
    const int64_t gate_dim = projection_dim / 2;
    return runtime_ev.ctx.temporal_layer_gating_in_ok &&
           runtime_ev.ctx.temporal_layer_index >= 0 &&
           runtime_ev.ctx.temporal_layer_index <
               runtime_ev.request.model.moshi_lm.num_layers &&
           gate_dim > 0 &&
           static_cast<uint64_t>(gate_dim) <= detail::k_max_gating_dim;
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
    const auto *linear_in = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, layer, "gating.linear_in.weight");
    const auto *linear_out = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, layer, "gating.linear_out.weight");
    int64_t projection_dim = 0;
    if (linear_in != nullptr && linear_in->n_dims > 1) {
      projection_dim = linear_in->dims[1];
    }
    const int64_t gate_dim = projection_dim / 2;
    return runtime_ev.ctx.temporal_layer_silu_gate_ok && layer >= 0 &&
           layer < runtime_ev.request.model.moshi_lm.num_layers &&
           hidden_dim > 0 && gate_dim > 0 &&
           static_cast<uint64_t>(gate_dim) <= detail::k_max_gating_dim &&
           detail::tensor_shape(linear_out, gate_dim, hidden_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(linear_out->type));
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
    const auto *out_norm =
        detail::find_tensor(runtime_ev.request.model, "lm.out_norm.alpha");
    return guard_temporal_layers_complete{}(runtime_ev, ctx) &&
           hidden_dim > 0 &&
           static_cast<uint64_t>(hidden_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(out_norm, hidden_dim) &&
           static_cast<detail::dtype>(out_norm->type) == detail::dtype::f32;
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
    const auto *text_linear =
        detail::find_tensor(runtime_ev.request.model, "lm.text_linear.weight");
    return runtime_ev.ctx.temporal_out_norm_ok && hidden_dim > 0 &&
           lm.text_card > 0 &&
           detail::tensor_shape(text_linear, hidden_dim, lm.text_card) &&
           detail::supported_argmax_dtype(
               static_cast<detail::dtype>(text_linear->type));
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
           detail::token_in_embedding_range(
               runtime_ev.request.forced_text_token,
               runtime_ev.request.model.moshi_lm.text_card);
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
    return ctx.sampling.enabled && ctx.sampling.text_temperature > 0.0f &&
           ctx.sampling.text_top_k > 0 &&
           static_cast<uint64_t>(ctx.session.text_card) <=
               detail::k_max_sampling_card &&
           static_cast<uint64_t>(ctx.sampling.text_top_k) <=
               detail::k_max_sampling_top_k;
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

struct guard_depformer_kv_binding_present {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    (void)runtime_ev;
    return ctx.depformer_kv.cache != nullptr &&
           ctx.depformer_kv.bind != nullptr;
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
           view.offset != nullptr && *view.offset >= 0 &&
           view.layer_count > 0 && view.position_capacity > 0 &&
           view.kv_dim > 0;
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
        static_cast<uint32_t>(codebook) >=
            lm.depformer_weight_schedule_count) {
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
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
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
    const auto *input_projection = detail::find_indexed_tensor(
        model, "lm.depformer_in.%d.weight", weight_index);
    const auto *embedding =
        detail::find_tensor(model, "lm.depformer_text_emb.weight");
    return runtime_ev.ctx.depformer_kv_bound && runtime_ev.ctx.text_logits_ok &&
           codebook == 0 && hidden_dim > 0 && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(input_projection, hidden_dim, dep_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(input_projection->type)) &&
           detail::tensor_shape(embedding, dep_dim, lm.text_card + 1) &&
           detail::supported_get_rows_dtype(
               static_cast<detail::dtype>(embedding->type)) &&
           detail::token_in_embedding_range(runtime_ev.request.text_token_out,
                                            lm.text_card);
  }
};

struct guard_depformer_audio_input_supported {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
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
    const auto *input_projection = detail::find_indexed_tensor(
        model, "lm.depformer_in.%d.weight", weight_index);
    const auto *embedding = detail::find_indexed_tensor(
        model, "lm.depformer_emb.%d.weight", codebook - 1);
    return runtime_ev.ctx.depformer_kv_bound && runtime_ev.ctx.text_logits_ok &&
           codebook > 0 && codebook < lm.dep_q && hidden_dim > 0 &&
           dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(input_projection, hidden_dim, dep_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(input_projection->type)) &&
           detail::tensor_shape(embedding, dep_dim, lm.card + 1) &&
           detail::supported_get_rows_dtype(
               static_cast<detail::dtype>(embedding->type)) &&
           detail::token_in_embedding_range(
               runtime_ev.request
                   .audio_tokens_out[static_cast<size_t>(codebook - 1)],
               lm.card);
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
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const auto *norm1 =
        detail::find_depformer_tensor(model, layer, "norm1.alpha");
    return runtime_ev.ctx.depformer_input_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(norm1, dep_dim) &&
           static_cast<detail::dtype>(norm1->type) == detail::dtype::f32;
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
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    if (weight_index < 0 || weight_index >= lm.dep_q) {
      return false;
    }
    const auto *projection =
        detail::find_depformer_projection(model, layer, weight_index);
    return runtime_ev.ctx.depformer_layer_norm_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) * 3u <= detail::k_max_qkv_dim &&
           detail::tensor_shape(projection, dep_dim, dep_dim * 3) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(projection->type));
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
    if (!runtime_ev.ctx.depformer_layer_projection_ok ||
        view.offset == nullptr || *view.offset < 0 || dep_dim <= 0 ||
        view.kv_dim != dep_dim || layer < 0 || layer >= view.layer_count ||
        view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size() ||
        view.position_capacity <= 0) {
      return false;
    }
    const int32_t physical = *view.offset % view.position_capacity;
    const size_t layer_offset =
        view.layer_cache_offsets[static_cast<size_t>(layer)];
    const size_t begin = layer_offset + static_cast<size_t>(physical) *
                                            static_cast<size_t>(dep_dim);
    const size_t end = begin + static_cast<size_t>(dep_dim);
    return end <= view.key_cache.size() && end <= view.value_cache.size();
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
                  const action::context &) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const auto &view = runtime_ev.ctx.depformer_kv;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t head_dim =
        lm.depformer_num_heads > 0 ? dep_dim / lm.depformer_num_heads : 0;
    const int32_t valid_positions = runtime_ev.ctx.depformer_valid_positions;
    const int32_t capacity = view.position_capacity;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    if (!runtime_ev.ctx.depformer_layer_cache_write_ok || dep_dim <= 0 ||
        lm.depformer_num_heads <= 0 || dep_dim % lm.depformer_num_heads != 0 ||
        head_dim <= 0 ||
        static_cast<uint64_t>(head_dim) > detail::k_max_hidden_dim ||
        valid_positions <= 0 || capacity <= 0 ||
        static_cast<uint64_t>(capacity) > detail::k_max_depformer_context ||
        valid_positions > capacity || view.kv_dim != dep_dim || layer < 0 ||
        layer >= view.layer_count || view.layer_cache_offsets.empty() ||
        static_cast<size_t>(layer) >= view.layer_cache_offsets.size()) {
      return false;
    }
    const size_t layer_offset =
        view.layer_cache_offsets[static_cast<size_t>(layer)];
    const size_t dim = static_cast<size_t>(dep_dim);
    for (int32_t physical = 0; physical < capacity; ++physical) {
      const size_t begin = layer_offset + static_cast<size_t>(physical) * dim;
      const size_t end = begin + dim;
      if (end > view.key_cache.size() || end > view.value_cache.size()) {
        return false;
      }
    }
    return true;
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
    const auto *projection = detail::find_depformer_codebook_tensor(
        model, layer, "self_attn.out_projs.%d.weight", weight_index);
    return runtime_ev.ctx.depformer_layer_attention_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0 &&
           detail::tensor_shape(projection, dep_dim, dep_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(projection->type));
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
                  const action::context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const auto *norm2 =
        detail::find_depformer_tensor(model, layer, "norm2.alpha");
    return runtime_ev.ctx.depformer_layer_residual_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && dep_dim > 0 &&
           static_cast<uint64_t>(dep_dim) <= detail::k_max_hidden_dim &&
           detail::tensor_shape(norm2, dep_dim) &&
           static_cast<detail::dtype>(norm2->type) == detail::dtype::f32;
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
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_in.weight", weight_index);
    int64_t projection_dim = 0;
    if (linear_in != nullptr && linear_in->n_dims > 1) {
      projection_dim = linear_in->dims[1];
    }
    return runtime_ev.ctx.depformer_layer_norm2_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0 &&
           projection_dim > 0 && (projection_dim % 2) == 0 &&
           static_cast<uint64_t>(projection_dim) <=
               detail::k_max_gating_projection_dim &&
           detail::tensor_shape(linear_in, dep_dim, projection_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(linear_in->type));
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
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        runtime_ev.request.model, runtime_ev.ctx.depformer_layer_index,
        "gating.%d.linear_in.weight", weight_index);
    int64_t projection_dim = 0;
    if (linear_in != nullptr && linear_in->n_dims > 1) {
      projection_dim = linear_in->dims[1];
    }
    const int64_t gate_dim = projection_dim / 2;
    return runtime_ev.ctx.depformer_layer_gating_in_ok &&
           runtime_ev.ctx.depformer_layer_index >= 0 &&
           runtime_ev.ctx.depformer_layer_index <
               runtime_ev.request.model.moshi_lm.depformer_num_layers &&
           gate_dim > 0 &&
           static_cast<uint64_t>(gate_dim) <= detail::k_max_gating_dim;
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
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_in.weight", weight_index);
    const auto *linear_out = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_out.weight", weight_index);
    int64_t projection_dim = 0;
    if (linear_in != nullptr && linear_in->n_dims > 1) {
      projection_dim = linear_in->dims[1];
    }
    const int64_t gate_dim = projection_dim / 2;
    return runtime_ev.ctx.depformer_layer_silu_gate_ok && layer >= 0 &&
           layer < model.moshi_lm.depformer_num_layers && codebook >= 0 &&
           codebook < model.moshi_lm.dep_q && dep_dim > 0 && gate_dim > 0 &&
           static_cast<uint64_t>(gate_dim) <= detail::k_max_gating_dim &&
           detail::tensor_shape(linear_out, gate_dim, dep_dim) &&
           detail::supported_mul_mat_dtype(
               static_cast<detail::dtype>(linear_out->type));
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
    const auto *linear =
        detail::find_indexed_tensor(model, "lm.linears.%d.weight", codebook);
    return guard_depformer_layers_complete{}(runtime_ev, ctx) &&
           codebook >= 0 && codebook < lm.dep_q && dep_dim > 0 && lm.card > 0 &&
           detail::tensor_shape(linear, dep_dim, lm.card) &&
           detail::supported_argmax_dtype(
               static_cast<detail::dtype>(linear->type));
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
    return ctx.sampling.enabled && ctx.sampling.audio_temperature > 0.0f &&
           ctx.sampling.audio_top_k > 0 &&
           static_cast<uint64_t>(ctx.session.audio_card) <=
               detail::k_max_sampling_card &&
           static_cast<uint64_t>(ctx.sampling.audio_top_k) <=
               detail::k_max_sampling_top_k;
  }
};

struct guard_depformer_sampling_config_invalid {
  bool operator()(const event::step_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return ctx.sampling.enabled && ctx.sampling.audio_temperature > 0.0f &&
           !guard_depformer_sampling_config_valid{}(runtime_ev, ctx);
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

} // namespace emel::speech::generator::moshi::executor::guard
