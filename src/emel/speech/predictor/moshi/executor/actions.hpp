#pragma once

#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/executor/context.hpp"
#include "emel/speech/predictor/moshi/executor/detail.hpp"
#include "emel/speech/predictor/moshi/executor/events.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace emel::speech::predictor::moshi::executor::action {

namespace detail_ns {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail_ns

struct effect_bind_contract {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.session = {};
    ctx.sampling = {};
    ctx.session.model = &runtime_ev.request.model;
    ctx.session.codebook_count = runtime_ev.request.model.moshi_lm.n_q + 1;
    ctx.session.dep_q = runtime_ev.request.model.moshi_lm.dep_q;
    ctx.session.text_card = runtime_ev.request.model.moshi_lm.text_card;
    ctx.session.audio_card = runtime_ev.request.model.moshi_lm.card;
    ctx.session.hidden_dim = runtime_ev.request.model.moshi_lm.dim;
    ctx.sampling.enabled = runtime_ev.request.sampling_enabled;
    ctx.sampling.consume_forced_text =
        runtime_ev.request.sampling_consume_forced_text;
    ctx.sampling.audio_temperature =
        runtime_ev.request.sampling_audio_temperature;
    ctx.sampling.text_temperature =
        runtime_ev.request.sampling_text_temperature;
    (void)emel::model::moshi::detail::build_execution_contract(
        runtime_ev.request.model, ctx.session.contract);
  }
};

struct effect_bind_nonzero_sampling_seed {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.sampling.random_state = runtime_ev.request.sampling_seed;
  }
};

struct effect_bind_zero_sampling_seed {
  void operator()(const event::initialize_run &, context &ctx) const noexcept {
    ctx.sampling.random_state = ctx.policy.zero_seed_state;
  }
};

struct effect_bind_requested_text_sampling_top_k {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.sampling.text_top_k = runtime_ev.request.sampling_text_top_k;
  }
};

struct effect_bind_full_card_text_sampling_top_k {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.sampling.text_top_k = runtime_ev.request.model.moshi_lm.text_card;
  }
};

struct effect_bind_requested_audio_sampling_top_k {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.sampling.audio_top_k = runtime_ev.request.sampling_audio_top_k;
  }
};

struct effect_bind_full_card_audio_sampling_top_k {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.sampling.audio_top_k = runtime_ev.request.model.moshi_lm.card;
  }
};

struct effect_mark_bind_failed {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::bind_failed);
  }
};

struct effect_mark_not_initialized {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::not_initialized);
  }
};

struct effect_mark_model_mismatch {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::model_mismatch);
  }
};

struct effect_mark_request_shape {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::request_shape);
  }
};

struct effect_mark_graph_execution_unsupported {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err =
        detail_ns::to_error(error::graph_execution_unsupported);
  }
};

struct effect_begin_input_embedding {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    (void)model;
    runtime_ev.ctx.input_embedding_ok = false;
    runtime_ev.ctx.input_text_embedding_ok = false;
    runtime_ev.ctx.input_audio_embedding_ok = false;
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.temporal_position_accepted = false;
    runtime_ev.ctx.temporal_position_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.temporal_position = {};
    runtime_ev.ctx.depformer_input_projection_bound = false;
    runtime_ev.ctx.depformer_input_projection_ok = false;
    runtime_ev.ctx.temporal_layer_norm_rms_ok = false;
    runtime_ev.ctx.temporal_layer_norm_ok = false;
    runtime_ev.ctx.temporal_layer_projection_ok = false;
    runtime_ev.ctx.temporal_layer_rope_ok = false;
    runtime_ev.ctx.temporal_layer_cache_write_ok = false;
    runtime_ev.ctx.temporal_layer_attention_ok = false;
    runtime_ev.ctx.temporal_layer_out_projection_ok = false;
    runtime_ev.ctx.temporal_layer_residual_ok = false;
    runtime_ev.ctx.temporal_layer_norm2_rms_ok = false;
    runtime_ev.ctx.temporal_layer_norm2_ok = false;
    runtime_ev.ctx.temporal_layer_gating_in_ok = false;
    runtime_ev.ctx.temporal_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.temporal_layer_silu_gate_ok = false;
    runtime_ev.ctx.temporal_layer_gating_out_ok = false;
    runtime_ev.ctx.temporal_layer_ff_residual_ok = false;
    runtime_ev.ctx.temporal_out_norm_rms_ok = false;
    runtime_ev.ctx.temporal_out_norm_ok = false;
    runtime_ev.ctx.text_logits_ok = false;
    runtime_ev.ctx.depformer_kv_bound = false;
    runtime_ev.ctx.depformer_position_accepted = false;
    runtime_ev.ctx.depformer_position_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.depformer_position = {};
    runtime_ev.ctx.depformer_input_ok = false;
    runtime_ev.ctx.depformer_layer_norm_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm_ok = false;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    runtime_ev.ctx.depformer_layer_cache_write_ok = false;
    runtime_ev.ctx.depformer_layer_attention_ok = false;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    runtime_ev.ctx.depformer_layer_residual_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    runtime_ev.ctx.depformer_layer_ff_residual_ok = false;
    runtime_ev.ctx.depformer_logits_ok = false;
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    runtime_ev.ctx.input_audio_codebook_index = 0;
    runtime_ev.ctx.temporal_layer_index = 0;
    runtime_ev.ctx.temporal_rope_position = -1;
    runtime_ev.ctx.depformer_codebook_index = 0;
    runtime_ev.ctx.depformer_weight_index = -1;
    runtime_ev.ctx.depformer_layer_index = 0;
    runtime_ev.ctx.embedding_dim = 0;
    runtime_ev.ctx.embedding_view = {};
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.hidden.data(), static_cast<size_t>(hidden_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.transformer_out.data(),
                static_cast<size_t>(hidden_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(),
                static_cast<size_t>(hidden_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.qkv.data(), static_cast<size_t>(hidden_dim) * 3u,
                0.0f);
    std::fill_n(runtime_ev.ctx.attention.data(),
                static_cast<size_t>(hidden_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.projection.data(),
                static_cast<size_t>(hidden_dim), 0.0f);
    std::fill(runtime_ev.ctx.gating_projection.begin(),
              runtime_ev.ctx.gating_projection.end(), 0.0f);
    std::fill(runtime_ev.ctx.gating_silu.begin(),
              runtime_ev.ctx.gating_silu.end(), 0.0f);
    std::fill(runtime_ev.ctx.gated.begin(), runtime_ev.ctx.gated.end(), 0.0f);
  }
};

struct effect_use_depformer_scheduled_weight {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    runtime_ev.ctx.depformer_weight_index =
        lm.depformer_weight_schedule[static_cast<size_t>(codebook)];
  }
};

struct effect_use_depformer_codebook_weight {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.ctx.depformer_weight_index =
        runtime_ev.ctx.depformer_codebook_index;
  }
};

struct effect_apply_external_input_embedding {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.input_embedding_ok = false;
    for (int32_t index = 0; index < hidden_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] =
          runtime_ev.request.input_embedding[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.input_embedding_ok = true;
  }
};

struct effect_bind_input_text_embedding {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const auto *text_emb =
        detail::find_tensor(runtime_ev.request.model, "lm.text_emb.weight");
    runtime_ev.ctx.embedding_view = {};
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.embedding_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.row_index = runtime_ev.request.input_sequence[0];
    runtime_ev.ctx.input_text_embedding_ok = false;
    runtime_ev.ctx.embedding_view_bound =
        detail::bind_tensor_view(*text_emb, runtime_ev.ctx.embedding_view);
  }
};

struct effect_run_embedding_row_fetch {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t embedding_dim = runtime_ev.ctx.embedding_dim;
    runtime_ev.ctx.embedding_row_ok = false;
    std::fill_n(runtime_ev.ctx.row.data(), static_cast<size_t>(embedding_dim),
                0.0f);
    emel::kernel::event::op_get_rows rows_ev{
        .src0 = runtime_ev.ctx.embedding_view,
        .src1 = detail::make_i32_src(&runtime_ev.ctx.row_index, 1u),
        .dst = detail::make_f32_dst(runtime_ev.ctx.row.data(),
                                    static_cast<uint64_t>(embedding_dim)),
    };
    runtime_ev.ctx.embedding_row_ok = ctx.kernel.process_event(rows_ev);
  }
};

struct effect_apply_input_text_embedding_row {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t embedding_dim = runtime_ev.ctx.embedding_dim;
    runtime_ev.ctx.input_text_embedding_ok = false;
    for (int32_t index = 0; index < embedding_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.row[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.input_text_embedding_ok = true;
  }
};

struct effect_bind_input_audio_embedding {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const int32_t index = runtime_ev.ctx.input_audio_codebook_index;
    const auto *audio_emb = detail::find_indexed_tensor(
        runtime_ev.request.model, "lm.emb.%d.weight", index);
    runtime_ev.ctx.embedding_view = {};
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.embedding_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.row_index =
        runtime_ev.request.input_sequence[static_cast<size_t>(index + 1)];
    runtime_ev.ctx.input_audio_embedding_ok = false;
    runtime_ev.ctx.embedding_view_bound =
        detail::bind_tensor_view(*audio_emb, runtime_ev.ctx.embedding_view);
  }
};

struct effect_apply_input_audio_embedding_row {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t embedding_dim = runtime_ev.ctx.embedding_dim;
    runtime_ev.ctx.input_audio_embedding_ok = false;
    for (int32_t index = 0; index < embedding_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.row[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.input_audio_embedding_ok = true;
  }
};

struct effect_skip_zero_input_audio_embedding {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.input_audio_embedding_ok = true;
  }
};

struct effect_advance_input_audio_codebook {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    ++runtime_ev.ctx.input_audio_codebook_index;
    runtime_ev.ctx.input_audio_embedding_ok = false;
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.embedding_dim = 0;
    runtime_ev.ctx.embedding_view = {};
  }
};

struct effect_finish_input_embedding {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.input_embedding_ok = true;
  }
};

struct effect_run_temporal_layer_norm_rms {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_norm_rms_ok = false;
    runtime_ev.ctx.temporal_layer_norm_ok = false;

    std::fill_n(runtime_ev.ctx.row.data(), static_cast<size_t>(hidden_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(),
                static_cast<size_t>(hidden_dim), 0.0f);

    emel::kernel::event::op_rms_norm rms_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.hidden.data(),
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.row.data(),
                                    static_cast<uint64_t>(hidden_dim)),
    };
    detail::set_op_param_f32(rms_ev, 0u, ctx.policy.rms_norm_epsilon);
    runtime_ev.ctx.temporal_layer_norm_rms_ok =
        ctx.kernel.process_event(rms_ev);
  }
};

struct effect_run_temporal_layer_norm_scale {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    runtime_ev.ctx.temporal_layer_norm_ok = false;

    const auto *norm1 =
        detail::find_lm_transformer_tensor(model, layer, "norm1.alpha");
    const auto *alpha = static_cast<const float *>(norm1->data);

    emel::kernel::event::op_mul mul_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.row.data(),
                                     static_cast<uint64_t>(hidden_dim)),
        .src1 = detail::make_f32_src(alpha, static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.normalized.data(),
                                    static_cast<uint64_t>(hidden_dim)),
    };
    runtime_ev.ctx.temporal_layer_norm_ok = ctx.kernel.process_event(mul_ev);
  }
};

struct effect_bind_temporal_layer_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    runtime_ev.ctx.temporal_layer_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.qkv.data(), static_cast<size_t>(hidden_dim) * 3u,
                0.0f);

    const auto *projection =
        detail::find_lm_transformer_projection(model, layer);
    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*projection, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_temporal_layer_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_projection_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.normalized.data(), 1u,
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.qkv.data(), 1u,
                                    static_cast<uint64_t>(hidden_dim) * 3u),
    };
    runtime_ev.ctx.temporal_layer_projection_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

template <uint64_t projection_part> struct effect_run_temporal_layer_rope {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t head_dim = hidden_dim / lm.num_heads;
    runtime_ev.ctx.temporal_layer_rope_ok = false;
    runtime_ev.ctx.temporal_rope_position =
        static_cast<int32_t>(runtime_ev.ctx.temporal_position.logical_position);

    const uint64_t head_dim_u64 = static_cast<uint64_t>(head_dim);
    const uint64_t heads_u64 = static_cast<uint64_t>(lm.num_heads);
    const uint64_t hidden_dim_u64 = static_cast<uint64_t>(hidden_dim);
    const uint64_t element_bytes = sizeof(float);
    const float *source =
        runtime_ev.ctx.qkv.data() + projection_part * hidden_dim_u64;
    emel::kernel::event::op_rope rope_ev{
        .src0 =
            {
                .data = source,
                .type = emel::kernel::event::dtype::f32,
                .ne = {head_dim_u64, heads_u64, 1u, 1u},
                .nb = {element_bytes, element_bytes * head_dim_u64,
                       element_bytes * hidden_dim_u64,
                       element_bytes * hidden_dim_u64},
            },
        .src1 =
            {
                .data = &runtime_ev.ctx.temporal_rope_position,
                .type = emel::kernel::event::dtype::i32,
                .ne = {1u, 1u, 1u, 1u},
                .nb = {sizeof(int32_t), sizeof(int32_t), sizeof(int32_t),
                       sizeof(int32_t)},
            },
        .dst =
            {
                .data = runtime_ev.ctx.rope.data(),
                .type = emel::kernel::event::dtype::f32,
                .ne = {head_dim_u64, heads_u64, 1u, 1u},
                .nb = {element_bytes, element_bytes * head_dim_u64,
                       element_bytes * hidden_dim_u64,
                       element_bytes * hidden_dim_u64},
            },
    };
    const int32_t rope_mode = emel::kernel::detail::rope_mode_timestep;
    const float freq_base = static_cast<float>(lm.max_period);
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;
    std::memcpy(rope_ev.op_params.data() + 1u * sizeof(int32_t), &head_dim,
                sizeof(head_dim));
    std::memcpy(rope_ev.op_params.data() + 2u * sizeof(int32_t), &rope_mode,
                sizeof(rope_mode));
    std::memcpy(rope_ev.op_params.data() + 5u * sizeof(float), &freq_base,
                sizeof(freq_base));
    std::memcpy(rope_ev.op_params.data() + 6u * sizeof(float), &freq_scale,
                sizeof(freq_scale));
    std::memcpy(rope_ev.op_params.data() + 7u * sizeof(float), &ext_factor,
                sizeof(ext_factor));
    std::memcpy(rope_ev.op_params.data() + 8u * sizeof(float), &attn_factor,
                sizeof(attn_factor));
    rope_ev.op_params_size = 9u * sizeof(float);
    runtime_ev.ctx.temporal_layer_rope_ok = ctx.kernel.process_event(rope_ev);
  }
};

using effect_run_temporal_layer_query_rope = effect_run_temporal_layer_rope<0u>;
using effect_run_temporal_layer_key_rope = effect_run_temporal_layer_rope<1u>;

template <uint64_t projection_part> struct effect_copy_temporal_layer_rope {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const size_t hidden_dim = static_cast<size_t>(ctx.session.hidden_dim);
    std::memcpy(runtime_ev.ctx.qkv.data() + projection_part * hidden_dim,
                runtime_ev.ctx.rope.data(), hidden_dim * sizeof(float));
  }
};

using effect_copy_temporal_layer_query_rope =
    effect_copy_temporal_layer_rope<0u>;
using effect_copy_temporal_layer_key_rope = effect_copy_temporal_layer_rope<1u>;

struct effect_write_temporal_layer_kv_cache {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_cache_write_ok = false;
    auto &view = runtime_ev.ctx.temporal_kv;
    const size_t layer_offset = view.layer_cache_offsets[static_cast<size_t>(
        runtime_ev.ctx.temporal_layer_index)];
    const size_t physical_position =
        static_cast<size_t>(runtime_ev.ctx.temporal_position.physical_position);
    const size_t dim = static_cast<size_t>(hidden_dim);
    const size_t begin = layer_offset + physical_position * dim;
    const float *key = runtime_ev.ctx.qkv.data() + dim;
    const float *value = runtime_ev.ctx.qkv.data() + dim * 2u;

    for (size_t index = 0; index < dim; ++index) {
      view.key_cache[begin + index] =
          emel::kernel::detail::fp32_to_bf16(key[index]);
      view.value_cache[begin + index] =
          emel::kernel::detail::fp32_to_bf16(value[index]);
    }
    runtime_ev.ctx.temporal_layer_cache_write_ok = true;
  }
};

struct effect_run_temporal_layer_attention {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const auto &view = runtime_ev.ctx.temporal_kv;
    const auto &window = runtime_ev.ctx.temporal_position.window;
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t head_dim = hidden_dim / lm.num_heads;
    const int32_t capacity = view.position_capacity;
    const int32_t valid_positions = window.valid_positions;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t layer_offset = view.layer_cache_offsets[static_cast<size_t>(
        runtime_ev.ctx.temporal_layer_index)];
    runtime_ev.ctx.temporal_layer_attention_ok = false;
    std::fill_n(runtime_ev.ctx.attention.data(),
                static_cast<size_t>(hidden_dim), 0.0f);

    for (int32_t head = 0; head < lm.num_heads; ++head) {
      const int32_t head_offset = head * head_dim;
      const float *q_head =
          runtime_ev.ctx.qkv.data() + static_cast<size_t>(head_offset);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        runtime_ev.ctx.q_bf16[static_cast<size_t>(dim)] =
            emel::kernel::detail::fp32_to_bf16(q_head[dim]);
      }
      for (int32_t physical = 0; physical < capacity; ++physical) {
        runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)] =
            -std::numeric_limits<float>::infinity();
      }
      for (int32_t position = 0; position < valid_positions; ++position) {
        const int32_t unwrapped = window.physical_begin + position;
        const int32_t physical =
            unwrapped - static_cast<int32_t>(unwrapped >= capacity) * capacity;
        const size_t cache_begin =
            layer_offset +
            static_cast<size_t>(physical) * static_cast<size_t>(hidden_dim) +
            static_cast<size_t>(head_offset);
        runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)] =
            emel::kernel::detail::vec_dot_bf16_ggml(
                head_dim, view.key_cache.data() + cache_begin,
                runtime_ev.ctx.q_bf16.data()) *
            scale;
      }

      emel::kernel::detail::soft_max_row_ggml(
          capacity, runtime_ev.ctx.attention_scores.data());
      for (int32_t physical = 0; physical < capacity; ++physical) {
        runtime_ev.ctx.attention_weights_bf16[static_cast<size_t>(physical)] =
            emel::kernel::detail::fp32_to_bf16(
                runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)]);
      }

      float *attention_head =
          runtime_ev.ctx.attention.data() + static_cast<size_t>(head_offset);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        double sum = 0.0;
        for (int32_t position = 0; position < valid_positions; ++position) {
          const int32_t unwrapped = window.physical_begin + position;
          const int32_t physical =
              unwrapped -
              static_cast<int32_t>(unwrapped >= capacity) * capacity;
          const size_t value_index =
              layer_offset +
              static_cast<size_t>(physical) * static_cast<size_t>(hidden_dim) +
              static_cast<size_t>(head_offset + dim);
          sum += static_cast<double>(
              emel::kernel::detail::bf16_to_fp32(
                  view.value_cache[value_index]) *
              emel::kernel::detail::bf16_to_fp32(
                  runtime_ev.ctx
                      .attention_weights_bf16[static_cast<size_t>(physical)]));
        }
        attention_head[dim] = static_cast<float>(sum);
      }
    }
    runtime_ev.ctx.temporal_layer_attention_ok = true;
  }
};

struct effect_bind_temporal_layer_out_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    runtime_ev.ctx.temporal_layer_out_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.projection.data(),
                static_cast<size_t>(hidden_dim), 0.0f);

    const auto *projection = detail::find_lm_transformer_tensor(
        model, layer, "self_attn.out_projs.0.weight");
    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*projection, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_temporal_layer_out_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_out_projection_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.attention.data(), 1u,
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.projection.data(), 1u,
                                    static_cast<uint64_t>(hidden_dim)),
    };
    runtime_ev.ctx.temporal_layer_out_projection_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_apply_temporal_layer_attention_residual {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_residual_ok = false;
    for (int32_t index = 0; index < hidden_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.projection[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.temporal_layer_residual_ok = true;
  }
};

struct effect_run_temporal_layer_norm2_rms {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_norm2_rms_ok = false;
    runtime_ev.ctx.temporal_layer_norm2_ok = false;

    std::fill_n(runtime_ev.ctx.row.data(), static_cast<size_t>(hidden_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(),
                static_cast<size_t>(hidden_dim), 0.0f);

    emel::kernel::event::op_rms_norm rms_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.hidden.data(),
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.row.data(),
                                    static_cast<uint64_t>(hidden_dim)),
    };
    detail::set_op_param_f32(rms_ev, 0u, ctx.policy.rms_norm_epsilon);
    runtime_ev.ctx.temporal_layer_norm2_rms_ok =
        ctx.kernel.process_event(rms_ev);
  }
};

struct effect_run_temporal_layer_norm2_scale {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    runtime_ev.ctx.temporal_layer_norm2_ok = false;

    const auto *norm2 =
        detail::find_lm_transformer_tensor(model, layer, "norm2.alpha");
    const auto *alpha = static_cast<const float *>(norm2->data);

    emel::kernel::event::op_mul mul_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.row.data(),
                                     static_cast<uint64_t>(hidden_dim)),
        .src1 = detail::make_f32_src(alpha, static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.normalized.data(),
                                    static_cast<uint64_t>(hidden_dim)),
    };
    runtime_ev.ctx.temporal_layer_norm2_ok = ctx.kernel.process_event(mul_ev);
  }
};

struct effect_bind_temporal_layer_gating_in {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    (void)ctx;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *linear_in = detail::find_lm_transformer_tensor(
        model, layer, "gating.linear_in.weight");
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    runtime_ev.ctx.temporal_layer_gating_in_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.gating_projection.data(),
                static_cast<size_t>(projection_dim), 0.0f);

    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*linear_in, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_temporal_layer_gating_in {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *linear_in = detail::find_lm_transformer_tensor(
        model, layer, "gating.linear_in.weight");
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    runtime_ev.ctx.temporal_layer_gating_in_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.normalized.data(), 1u,
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.gating_projection.data(), 1u,
                                    projection_dim),
    };
    runtime_ev.ctx.temporal_layer_gating_in_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_run_temporal_layer_silu_gate_silu {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto *linear_in = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, runtime_ev.ctx.temporal_layer_index,
        "gating.linear_in.weight");
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    const uint64_t gate_dim = projection_dim / 2u;
    runtime_ev.ctx.temporal_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.temporal_layer_silu_gate_ok = false;
    std::fill_n(runtime_ev.ctx.gating_silu.data(),
                static_cast<size_t>(gate_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.gated.data(), static_cast<size_t>(gate_dim),
                0.0f);

    emel::kernel::event::op_unary silu_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.gating_projection.data(),
                                     gate_dim),
        .dst =
            detail::make_f32_dst(runtime_ev.ctx.gating_silu.data(), gate_dim),
        .subop = emel::kernel::event::unary_subop::silu,
    };
    runtime_ev.ctx.temporal_layer_silu_gate_silu_ok =
        ctx.kernel.process_event(silu_ev);
  }
};

struct effect_run_temporal_layer_silu_gate_mul {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto *linear_in = detail::find_lm_transformer_tensor(
        runtime_ev.request.model, runtime_ev.ctx.temporal_layer_index,
        "gating.linear_in.weight");
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    const uint64_t gate_dim = projection_dim / 2u;
    const float *right = runtime_ev.ctx.gating_projection.data() + gate_dim;
    runtime_ev.ctx.temporal_layer_silu_gate_ok = false;

    emel::kernel::event::op_mul gate_ev{
        .src0 =
            detail::make_f32_src(runtime_ev.ctx.gating_silu.data(), gate_dim),
        .src1 = detail::make_f32_src(right, gate_dim),
        .dst = detail::make_f32_dst(runtime_ev.ctx.gated.data(), gate_dim),
    };
    runtime_ev.ctx.temporal_layer_silu_gate_ok =
        ctx.kernel.process_event(gate_ev);
  }
};

struct effect_bind_temporal_layer_gating_out {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *linear_in = detail::find_lm_transformer_tensor(
        model, layer, "gating.linear_in.weight");
    const auto *linear_out = detail::find_lm_transformer_tensor(
        model, layer, "gating.linear_out.weight");
    const uint64_t gate_dim = static_cast<uint64_t>(linear_in->dims[1]) / 2u;
    (void)gate_dim;
    runtime_ev.ctx.temporal_layer_gating_out_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.projection.data(),
                static_cast<size_t>(hidden_dim), 0.0f);

    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*linear_out, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_temporal_layer_gating_out {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t layer = runtime_ev.ctx.temporal_layer_index;
    const auto *linear_in = detail::find_lm_transformer_tensor(
        model, layer, "gating.linear_in.weight");
    const uint64_t gate_dim = static_cast<uint64_t>(linear_in->dims[1]) / 2u;
    runtime_ev.ctx.temporal_layer_gating_out_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.gated.data(), 1u, gate_dim),
        .dst = detail::make_f32_dst(runtime_ev.ctx.projection.data(), 1u,
                                    static_cast<uint64_t>(hidden_dim)),
    };
    runtime_ev.ctx.temporal_layer_gating_out_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_apply_temporal_layer_ff_residual {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_ff_residual_ok = false;
    for (int32_t index = 0; index < hidden_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.projection[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.temporal_layer_ff_residual_ok = true;
  }
};

struct effect_advance_temporal_layer {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    ++runtime_ev.ctx.temporal_layer_index;
    runtime_ev.ctx.temporal_layer_norm_rms_ok = false;
    runtime_ev.ctx.temporal_layer_norm_ok = false;
    runtime_ev.ctx.temporal_layer_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.temporal_layer_rope_ok = false;
    runtime_ev.ctx.temporal_layer_cache_write_ok = false;
    runtime_ev.ctx.temporal_layer_attention_ok = false;
    runtime_ev.ctx.temporal_layer_out_projection_ok = false;
    runtime_ev.ctx.temporal_layer_residual_ok = false;
    runtime_ev.ctx.temporal_layer_norm2_rms_ok = false;
    runtime_ev.ctx.temporal_layer_norm2_ok = false;
    runtime_ev.ctx.temporal_layer_gating_in_ok = false;
    runtime_ev.ctx.temporal_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.temporal_layer_silu_gate_ok = false;
    runtime_ev.ctx.temporal_layer_gating_out_ok = false;
    runtime_ev.ctx.temporal_layer_ff_residual_ok = false;
  }
};

struct effect_run_temporal_out_norm_rms {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_out_norm_rms_ok = false;
    runtime_ev.ctx.temporal_out_norm_ok = false;

    std::fill_n(runtime_ev.ctx.row.data(), static_cast<size_t>(hidden_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(),
                static_cast<size_t>(hidden_dim), 0.0f);

    emel::kernel::event::op_rms_norm rms_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.hidden.data(),
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.row.data(),
                                    static_cast<uint64_t>(hidden_dim)),
    };
    detail::set_op_param_f32(rms_ev, 0u, ctx.policy.rms_norm_epsilon);
    runtime_ev.ctx.temporal_out_norm_rms_ok = ctx.kernel.process_event(rms_ev);
  }
};

struct effect_run_temporal_out_norm_scale {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const auto *out_norm =
        detail::find_tensor(runtime_ev.request.model, "lm.out_norm.alpha");
    const auto *alpha = static_cast<const float *>(out_norm->data);
    runtime_ev.ctx.temporal_out_norm_ok = false;

    emel::kernel::event::op_mul mul_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.row.data(),
                                     static_cast<uint64_t>(hidden_dim)),
        .src1 = detail::make_f32_src(alpha, static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.normalized.data(),
                                    static_cast<uint64_t>(hidden_dim)),
    };
    runtime_ev.ctx.temporal_out_norm_ok = ctx.kernel.process_event(mul_ev);
  }
};

struct effect_publish_temporal_out_norm {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    std::copy_n(runtime_ev.ctx.normalized.data(),
                static_cast<size_t>(hidden_dim),
                runtime_ev.ctx.transformer_out.data());
  }
};

struct effect_bind_text_token_logits {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const auto *text_linear =
        detail::find_tensor(runtime_ev.request.model, "lm.text_linear.weight");
    runtime_ev.ctx.text_logits_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*text_linear, runtime_ev.ctx.projection_view);
  }
};

struct effect_select_text_token {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.text_logits_ok = false;
    emel::kernel::event::op_mul_mat_argmax logits_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.normalized.data(), 1u,
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(&runtime_ev.ctx.best_score, 1u, 1u),
        .index_out = &runtime_ev.ctx.best_index,
    };
    runtime_ev.ctx.text_logits_ok = ctx.kernel.process_event(logits_ev);
    runtime_ev.request.text_token_out = runtime_ev.ctx.best_index;
  }
};

struct effect_compute_text_token_logits {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t text_card = ctx.session.text_card;
    runtime_ev.ctx.text_logits_ok = false;
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    emel::kernel::event::op_mul_mat logits_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.normalized.data(), 1u,
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.logits.data(), 1u,
                                    static_cast<uint64_t>(text_card)),
    };
    runtime_ev.ctx.text_logits_ok = ctx.kernel.process_event(logits_ev);
  }
};

struct effect_select_text_sampling_token {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.sampler_error =
        emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::event::sample_temperature_top_k sample_ev{
        runtime_ev.ctx.logits,           ctx.session.text_card,
        ctx.sampling.text_temperature,   ctx.sampling.text_top_k,
        runtime_ev.ctx.sampling_indices, runtime_ev.ctx.top_scores,
        runtime_ev.ctx.top_indices,      ctx.sampling.random_state,
        runtime_ev.ctx.best_index,       runtime_ev.ctx.best_score,
        runtime_ev.ctx.sampler_error};
    runtime_ev.ctx.sampler_accepted = ctx.sampler->process_event(sample_ev);
  }
};

struct effect_publish_sampled_text_token {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.request.text_token_out = runtime_ev.ctx.best_index;
  }
};

struct effect_publish_forced_text_token {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    runtime_ev.ctx.text_logits_ok = true;
    runtime_ev.ctx.best_index = runtime_ev.request.forced_text_token;
    runtime_ev.request.text_token_out = runtime_ev.request.forced_text_token;
  }
};

struct effect_bind_depformer_kv {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.depformer_kv = ctx.depformer_kv;
    runtime_ev.ctx.depformer_kv_bound = true;
  }
};

struct effect_reset_depformer_positions {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.depformer_position_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.depformer_position_accepted =
        ctx.depformer_positions->process_event(
            emel::memory::streaming::event::reset{
                .error_out = runtime_ev.ctx.depformer_position_error});
  }
};

struct effect_advance_depformer_position {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.depformer_position = {};
    runtime_ev.ctx.depformer_position_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.depformer_position_accepted =
        ctx.depformer_positions->process_event(
            emel::memory::streaming::event::advance{
                .result = runtime_ev.ctx.depformer_position,
                .error_out = runtime_ev.ctx.depformer_position_error});
  }
};

struct effect_bind_depformer_text_input_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    (void)ctx;
    const int32_t dep_dim = lm.depformer_dim;
    runtime_ev.ctx.depformer_input_ok = false;
    runtime_ev.ctx.depformer_input_projection_bound = false;
    runtime_ev.ctx.depformer_input_projection_ok = false;
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.depformer_layer_index = 0;
    runtime_ev.ctx.depformer_layer_norm_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm_ok = false;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    runtime_ev.ctx.depformer_layer_cache_write_ok = false;
    runtime_ev.ctx.depformer_layer_attention_ok = false;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    runtime_ev.ctx.depformer_layer_residual_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    runtime_ev.ctx.depformer_layer_ff_residual_ok = false;
    runtime_ev.ctx.depformer_logits_ok = false;
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    std::fill_n(runtime_ev.ctx.hidden.data(), static_cast<size_t>(dep_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(), static_cast<size_t>(dep_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.projection.data(), static_cast<size_t>(dep_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.qkv.data(), static_cast<size_t>(dep_dim) * 3u,
                0.0f);
    std::fill_n(runtime_ev.ctx.attention.data(), static_cast<size_t>(dep_dim),
                0.0f);

    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    const auto *input_projection = detail::find_indexed_tensor(
        model, "lm.depformer_in.%d.weight", weight_index);
    runtime_ev.ctx.projection_view = {};
    runtime_ev.ctx.depformer_input_projection_bound = detail::bind_tensor_view(
        *input_projection, runtime_ev.ctx.projection_view);
  }
};

struct effect_bind_depformer_audio_input_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    (void)ctx;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    runtime_ev.ctx.depformer_input_ok = false;
    runtime_ev.ctx.depformer_input_projection_bound = false;
    runtime_ev.ctx.depformer_input_projection_ok = false;
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.depformer_layer_index = 0;
    runtime_ev.ctx.depformer_layer_norm_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm_ok = false;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    runtime_ev.ctx.depformer_layer_cache_write_ok = false;
    runtime_ev.ctx.depformer_layer_attention_ok = false;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    runtime_ev.ctx.depformer_layer_residual_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    runtime_ev.ctx.depformer_layer_ff_residual_ok = false;
    runtime_ev.ctx.depformer_logits_ok = false;
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    std::fill_n(runtime_ev.ctx.hidden.data(), static_cast<size_t>(dep_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(), static_cast<size_t>(dep_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.projection.data(), static_cast<size_t>(dep_dim),
                0.0f);
    std::fill_n(runtime_ev.ctx.qkv.data(), static_cast<size_t>(dep_dim) * 3u,
                0.0f);
    std::fill_n(runtime_ev.ctx.attention.data(), static_cast<size_t>(dep_dim),
                0.0f);

    const auto *input_projection = detail::find_indexed_tensor(
        model, "lm.depformer_in.%d.weight", weight_index);
    runtime_ev.ctx.projection_view = {};
    runtime_ev.ctx.depformer_input_projection_bound = detail::bind_tensor_view(
        *input_projection, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_depformer_input_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t dep_dim = lm.depformer_dim;
    runtime_ev.ctx.depformer_input_projection_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.transformer_out.data(), 1u,
                                     static_cast<uint64_t>(hidden_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.hidden.data(), 1u,
                                    static_cast<uint64_t>(dep_dim)),
    };
    runtime_ev.ctx.depformer_input_projection_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_bind_depformer_text_input_embedding {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    const auto *text_emb = detail::find_tensor(runtime_ev.request.model,
                                               "lm.depformer_text_emb.weight");
    runtime_ev.ctx.embedding_view = {};
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.embedding_dim = dep_dim;
    runtime_ev.ctx.row_index = runtime_ev.request.text_token_out;
    runtime_ev.ctx.depformer_input_ok = false;
    runtime_ev.ctx.embedding_view_bound =
        detail::bind_tensor_view(*text_emb, runtime_ev.ctx.embedding_view);
  }
};

struct effect_bind_depformer_audio_input_embedding {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;

    const auto *audio_emb = detail::find_indexed_tensor(
        runtime_ev.request.model, "lm.depformer_emb.%d.weight", codebook - 1);
    runtime_ev.ctx.embedding_view = {};
    runtime_ev.ctx.embedding_view_bound = false;
    runtime_ev.ctx.embedding_row_ok = false;
    runtime_ev.ctx.embedding_dim = dep_dim;
    runtime_ev.ctx.row_index =
        runtime_ev.request.audio_tokens_out[static_cast<size_t>(codebook - 1)];
    runtime_ev.ctx.depformer_input_ok = false;
    runtime_ev.ctx.embedding_view_bound =
        detail::bind_tensor_view(*audio_emb, runtime_ev.ctx.embedding_view);
  }
};

struct effect_apply_depformer_input_embedding_row {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t embedding_dim = runtime_ev.ctx.embedding_dim;
    runtime_ev.ctx.depformer_input_ok = false;
    for (int32_t index = 0; index < embedding_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.row[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.depformer_input_ok = true;
  }
};

struct effect_run_depformer_layer_norm_rms {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    runtime_ev.ctx.depformer_layer_norm_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm_ok = false;

    std::fill_n(runtime_ev.ctx.row.data(), static_cast<size_t>(dep_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(), static_cast<size_t>(dep_dim),
                0.0f);

    emel::kernel::event::op_rms_norm rms_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.hidden.data(),
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.row.data(),
                                    static_cast<uint64_t>(dep_dim)),
    };
    detail::set_op_param_f32(rms_ev, 0u, ctx.policy.rms_norm_epsilon);
    runtime_ev.ctx.depformer_layer_norm_rms_ok =
        ctx.kernel.process_event(rms_ev);
  }
};

struct effect_run_depformer_layer_norm_scale {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    runtime_ev.ctx.depformer_layer_norm_ok = false;

    const auto *norm1 =
        detail::find_depformer_tensor(model, layer, "norm1.alpha");
    const auto *alpha = static_cast<const float *>(norm1->data);

    emel::kernel::event::op_mul mul_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.row.data(),
                                     static_cast<uint64_t>(dep_dim)),
        .src1 = detail::make_f32_src(alpha, static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.normalized.data(),
                                    static_cast<uint64_t>(dep_dim)),
    };
    runtime_ev.ctx.depformer_layer_norm_ok = ctx.kernel.process_event(mul_ev);
  }
};

struct effect_bind_depformer_layer_projection {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.qkv.data(), static_cast<size_t>(dep_dim) * 3u,
                0.0f);

    const auto *projection =
        detail::find_depformer_projection(model, layer, weight_index);
    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*projection, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_depformer_layer_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.normalized.data(), 1u,
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.qkv.data(), 1u,
                                    static_cast<uint64_t>(dep_dim) * 3u),
    };
    runtime_ev.ctx.depformer_layer_projection_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_write_depformer_layer_kv_cache {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    auto &view = runtime_ev.ctx.depformer_kv;
    runtime_ev.ctx.depformer_layer_cache_write_ok = false;

    const int32_t physical_position =
        runtime_ev.ctx.depformer_position.physical_position;

    const size_t layer_offset = view.layer_cache_offsets[static_cast<size_t>(
        runtime_ev.ctx.depformer_layer_index)];
    const size_t begin = layer_offset + static_cast<size_t>(physical_position) *
                                            static_cast<size_t>(dep_dim);
    const float *key = runtime_ev.ctx.qkv.data() + static_cast<size_t>(dep_dim);
    const float *value =
        runtime_ev.ctx.qkv.data() + static_cast<size_t>(dep_dim) * 2u;

    for (int32_t index = 0; index < dep_dim; ++index) {
      view.key_cache[begin + static_cast<size_t>(index)] =
          emel::kernel::detail::fp32_to_bf16(key[index]);
      view.value_cache[begin + static_cast<size_t>(index)] =
          emel::kernel::detail::fp32_to_bf16(value[index]);
    }
    runtime_ev.ctx.depformer_layer_cache_write_ok = true;
  }
};

struct effect_run_depformer_layer_attention {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const auto &view = runtime_ev.ctx.depformer_kv;
    const auto &window = runtime_ev.ctx.depformer_position.window;
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t head_dim = dep_dim / lm.depformer_num_heads;
    const int32_t capacity = view.position_capacity;
    const int32_t valid_positions = window.valid_positions;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t layer_offset = view.layer_cache_offsets[static_cast<size_t>(
        runtime_ev.ctx.depformer_layer_index)];
    runtime_ev.ctx.depformer_layer_attention_ok = false;
    std::fill_n(runtime_ev.ctx.attention.data(), static_cast<size_t>(dep_dim),
                0.0f);

    for (int32_t head = 0; head < lm.depformer_num_heads; ++head) {
      const int32_t head_offset = head * head_dim;
      const float *q_head =
          runtime_ev.ctx.qkv.data() + static_cast<size_t>(head_offset);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        runtime_ev.ctx.q_bf16[static_cast<size_t>(dim)] =
            emel::kernel::detail::fp32_to_bf16(q_head[dim]);
      }
      for (int32_t physical = 0; physical < capacity; ++physical) {
        runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)] =
            -std::numeric_limits<float>::infinity();
      }
      for (int32_t position = 0; position < valid_positions; ++position) {
        const int32_t unwrapped = window.physical_begin + position;
        const int32_t physical =
            unwrapped - static_cast<int32_t>(unwrapped >= capacity) * capacity;
        const size_t cache_begin =
            layer_offset +
            static_cast<size_t>(physical) * static_cast<size_t>(dep_dim) +
            static_cast<size_t>(head_offset);
        runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)] =
            emel::kernel::detail::vec_dot_bf16_ggml(
                head_dim, view.key_cache.data() + cache_begin,
                runtime_ev.ctx.q_bf16.data()) *
            scale;
      }

      emel::kernel::detail::soft_max_row_ggml(
          capacity, runtime_ev.ctx.attention_scores.data());
      for (int32_t physical = 0; physical < capacity; ++physical) {
        runtime_ev.ctx.attention_weights_bf16[static_cast<size_t>(physical)] =
            emel::kernel::detail::fp32_to_bf16(
                runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)]);
      }

      float *attention_head =
          runtime_ev.ctx.attention.data() + static_cast<size_t>(head_offset);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        double sum = 0.0;
        for (int32_t position = 0; position < valid_positions; ++position) {
          const int32_t unwrapped = window.physical_begin + position;
          const int32_t physical =
              unwrapped -
              static_cast<int32_t>(unwrapped >= capacity) * capacity;
          const size_t value_index =
              layer_offset +
              static_cast<size_t>(physical) * static_cast<size_t>(dep_dim) +
              static_cast<size_t>(head_offset + dim);
          sum += static_cast<double>(
              emel::kernel::detail::bf16_to_fp32(
                  view.value_cache[value_index]) *
              emel::kernel::detail::bf16_to_fp32(
                  runtime_ev.ctx
                      .attention_weights_bf16[static_cast<size_t>(physical)]));
        }
        attention_head[dim] = static_cast<float>(sum);
      }
    }
    runtime_ev.ctx.depformer_layer_attention_ok = true;
  }
};

struct effect_bind_depformer_layer_out_projection {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.projection.data(), static_cast<size_t>(dep_dim),
                0.0f);

    const auto *projection = detail::find_depformer_codebook_tensor(
        model, layer, "self_attn.out_projs.%d.weight", weight_index);
    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*projection, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_depformer_layer_out_projection {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.attention.data(), 1u,
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.projection.data(), 1u,
                                    static_cast<uint64_t>(dep_dim)),
    };
    runtime_ev.ctx.depformer_layer_out_projection_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_apply_depformer_layer_attention_residual {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    runtime_ev.ctx.depformer_layer_residual_ok = false;
    for (int32_t index = 0; index < dep_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.projection[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.depformer_layer_residual_ok = true;
  }
};

struct effect_run_depformer_layer_norm2_rms {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    runtime_ev.ctx.depformer_layer_norm2_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;

    std::fill_n(runtime_ev.ctx.row.data(), static_cast<size_t>(dep_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.normalized.data(), static_cast<size_t>(dep_dim),
                0.0f);

    emel::kernel::event::op_rms_norm rms_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.hidden.data(),
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.row.data(),
                                    static_cast<uint64_t>(dep_dim)),
    };
    detail::set_op_param_f32(rms_ev, 0u, ctx.policy.rms_norm_epsilon);
    runtime_ev.ctx.depformer_layer_norm2_rms_ok =
        ctx.kernel.process_event(rms_ev);
  }
};

struct effect_run_depformer_layer_norm2_scale {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;

    const auto *norm2 =
        detail::find_depformer_tensor(model, layer, "norm2.alpha");
    const auto *alpha = static_cast<const float *>(norm2->data);

    emel::kernel::event::op_mul mul_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.row.data(),
                                     static_cast<uint64_t>(dep_dim)),
        .src1 = detail::make_f32_src(alpha, static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.normalized.data(),
                                    static_cast<uint64_t>(dep_dim)),
    };
    runtime_ev.ctx.depformer_layer_norm2_ok = ctx.kernel.process_event(mul_ev);
  }
};

struct effect_bind_depformer_layer_gating_in {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_in.weight", weight_index);
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.gating_projection.data(),
                static_cast<size_t>(projection_dim), 0.0f);

    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*linear_in, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_depformer_layer_gating_in {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const int32_t dep_dim = model.moshi_lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_in.weight", weight_index);
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.normalized.data(), 1u,
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.gating_projection.data(), 1u,
                                    projection_dim),
    };
    runtime_ev.ctx.depformer_layer_gating_in_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_run_depformer_layer_silu_gate_silu {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        runtime_ev.request.model, runtime_ev.ctx.depformer_layer_index,
        "gating.%d.linear_in.weight", runtime_ev.ctx.depformer_weight_index);
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    const uint64_t gate_dim = projection_dim / 2u;
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;
    std::fill_n(runtime_ev.ctx.gating_silu.data(),
                static_cast<size_t>(gate_dim), 0.0f);
    std::fill_n(runtime_ev.ctx.gated.data(), static_cast<size_t>(gate_dim),
                0.0f);

    emel::kernel::event::op_unary silu_ev{
        .src0 = detail::make_f32_src(runtime_ev.ctx.gating_projection.data(),
                                     gate_dim),
        .dst =
            detail::make_f32_dst(runtime_ev.ctx.gating_silu.data(), gate_dim),
        .subop = emel::kernel::event::unary_subop::silu,
    };
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok =
        ctx.kernel.process_event(silu_ev);
  }
};

struct effect_run_depformer_layer_silu_gate_mul {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        runtime_ev.request.model, runtime_ev.ctx.depformer_layer_index,
        "gating.%d.linear_in.weight", runtime_ev.ctx.depformer_weight_index);
    const uint64_t projection_dim = static_cast<uint64_t>(linear_in->dims[1]);
    const uint64_t gate_dim = projection_dim / 2u;
    const float *right = runtime_ev.ctx.gating_projection.data() + gate_dim;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;

    emel::kernel::event::op_mul gate_ev{
        .src0 =
            detail::make_f32_src(runtime_ev.ctx.gating_silu.data(), gate_dim),
        .src1 = detail::make_f32_src(right, gate_dim),
        .dst = detail::make_f32_dst(runtime_ev.ctx.gated.data(), gate_dim),
    };
    runtime_ev.ctx.depformer_layer_silu_gate_ok =
        ctx.kernel.process_event(gate_ev);
  }
};

struct effect_bind_depformer_layer_gating_out {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_in.weight", weight_index);
    const auto *linear_out = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_out.weight", weight_index);
    const uint64_t gate_dim = static_cast<uint64_t>(linear_in->dims[1]) / 2u;
    (void)gate_dim;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    std::fill_n(runtime_ev.ctx.projection.data(), static_cast<size_t>(dep_dim),
                0.0f);

    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*linear_out, runtime_ev.ctx.projection_view);
  }
};

struct effect_run_depformer_layer_gating_out {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t layer = runtime_ev.ctx.depformer_layer_index;
    const int32_t weight_index = runtime_ev.ctx.depformer_weight_index;
    const auto *linear_in = detail::find_depformer_codebook_tensor(
        model, layer, "gating.%d.linear_in.weight", weight_index);
    const uint64_t gate_dim = static_cast<uint64_t>(linear_in->dims[1]) / 2u;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    emel::kernel::event::op_mul_mat projection_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.gated.data(), 1u, gate_dim),
        .dst = detail::make_f32_dst(runtime_ev.ctx.projection.data(), 1u,
                                    static_cast<uint64_t>(dep_dim)),
    };
    runtime_ev.ctx.depformer_layer_gating_out_ok =
        ctx.kernel.process_event(projection_ev);
  }
};

struct effect_apply_depformer_layer_ff_residual {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    const int32_t dep_dim = runtime_ev.request.model.moshi_lm.depformer_dim;
    runtime_ev.ctx.depformer_layer_ff_residual_ok = false;
    for (int32_t index = 0; index < dep_dim; ++index) {
      runtime_ev.ctx.hidden[static_cast<size_t>(index)] +=
          runtime_ev.ctx.projection[static_cast<size_t>(index)];
    }
    runtime_ev.ctx.depformer_layer_ff_residual_ok = true;
  }
};

struct effect_advance_depformer_layer {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    ++runtime_ev.ctx.depformer_layer_index;
    runtime_ev.ctx.depformer_layer_norm_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm_ok = false;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.depformer_layer_cache_write_ok = false;
    runtime_ev.ctx.depformer_layer_attention_ok = false;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    runtime_ev.ctx.depformer_layer_residual_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    runtime_ev.ctx.depformer_layer_ff_residual_ok = false;
  }
};

struct effect_bind_depformer_token_logits {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    const auto *linear = detail::find_indexed_tensor(
        runtime_ev.request.model, "lm.linears.%d.weight", codebook);
    runtime_ev.ctx.depformer_logits_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.projection_view = {};
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    runtime_ev.ctx.projection_view_bound =
        detail::bind_tensor_view(*linear, runtime_ev.ctx.projection_view);
  }
};

struct effect_select_depformer_token {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    runtime_ev.ctx.depformer_logits_ok = false;
    emel::kernel::event::op_mul_mat_argmax logits_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.hidden.data(), 1u,
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(&runtime_ev.ctx.best_score, 1u, 1u),
        .index_out = &runtime_ev.ctx.best_index,
    };
    runtime_ev.ctx.depformer_logits_ok = ctx.kernel.process_event(logits_ev);
  }
};

struct effect_compute_depformer_token_logits {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t audio_card = ctx.session.audio_card;
    runtime_ev.ctx.depformer_logits_ok = false;
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    emel::kernel::event::op_mul_mat logits_ev{
        .src0 = runtime_ev.ctx.projection_view,
        .src1 = detail::make_f32_src(runtime_ev.ctx.hidden.data(), 1u,
                                     static_cast<uint64_t>(dep_dim)),
        .dst = detail::make_f32_dst(runtime_ev.ctx.logits.data(), 1u,
                                    static_cast<uint64_t>(audio_card)),
    };
    runtime_ev.ctx.depformer_logits_ok = ctx.kernel.process_event(logits_ev);
  }
};

struct effect_select_depformer_sampling_token {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.sampler_error =
        emel::error::cast(emel::logits::sampler::error::none);
    emel::logits::sampler::event::sample_temperature_top_k sample_ev{
        runtime_ev.ctx.logits,           ctx.session.audio_card,
        ctx.sampling.audio_temperature,  ctx.sampling.audio_top_k,
        runtime_ev.ctx.sampling_indices, runtime_ev.ctx.top_scores,
        runtime_ev.ctx.top_indices,      ctx.sampling.random_state,
        runtime_ev.ctx.best_index,       runtime_ev.ctx.best_score,
        runtime_ev.ctx.sampler_error};
    runtime_ev.ctx.sampler_accepted = ctx.sampler->process_event(sample_ev);
  }
};

struct effect_publish_depformer_token {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    runtime_ev.request.audio_tokens_out[static_cast<size_t>(codebook)] =
        runtime_ev.ctx.best_index;
  }
};

struct effect_advance_depformer_codebook {
  void operator()(const event::step_run &runtime_ev,
                  const context &) const noexcept {
    ++runtime_ev.ctx.depformer_codebook_index;
    runtime_ev.ctx.depformer_input_ok = false;
    runtime_ev.ctx.depformer_layer_index = 0;
    runtime_ev.ctx.depformer_layer_norm_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm_ok = false;
    runtime_ev.ctx.depformer_layer_projection_ok = false;
    runtime_ev.ctx.projection_view_bound = false;
    runtime_ev.ctx.depformer_layer_cache_write_ok = false;
    runtime_ev.ctx.depformer_layer_attention_ok = false;
    runtime_ev.ctx.depformer_layer_out_projection_ok = false;
    runtime_ev.ctx.depformer_layer_residual_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_rms_ok = false;
    runtime_ev.ctx.depformer_layer_norm2_ok = false;
    runtime_ev.ctx.depformer_layer_gating_in_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_silu_ok = false;
    runtime_ev.ctx.depformer_layer_silu_gate_ok = false;
    runtime_ev.ctx.depformer_layer_gating_out_ok = false;
    runtime_ev.ctx.depformer_layer_ff_residual_ok = false;
    runtime_ev.ctx.depformer_logits_ok = false;
    runtime_ev.ctx.depformer_weight_index = -1;
  }
};

struct effect_bind_temporal_kv {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.temporal_kv = ctx.temporal_kv;
    runtime_ev.ctx.temporal_kv_bound = true;
  }
};

struct effect_advance_temporal_position {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.temporal_position = {};
    runtime_ev.ctx.temporal_position_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.temporal_position_accepted =
        ctx.temporal_positions->process_event(
            emel::memory::streaming::event::advance{
                .result = runtime_ev.ctx.temporal_position,
                .error_out = runtime_ev.ctx.temporal_position_error});
  }
};

template <class runtime_event_type> struct effect_store_error_out {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.request.on_done(events::initialize_done{
        .request = &runtime_ev.request,
        .n_q = ctx.session.codebook_count - 1,
        .dep_q = ctx.session.dep_q,
    });
  }
};

struct effect_emit_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::initialize_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_reset_session {
  void operator()(const event::reset &, context &ctx) const noexcept {
    ctx.session = {};
  }
};

struct effect_mark_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = detail_ns::to_error(error::unexpected_event);
    }
  }
};

struct effect_mark_unexpected_and_store {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &ev,
                  context &ctx) const noexcept {
    effect_mark_unexpected{}(ev, ctx);
    if constexpr (requires {
                    ev.ctx.err;
                    ev.request.error_out;
                  }) {
      *ev.request.error_out = ev.ctx.err;
    }
  }
};

} // namespace emel::speech::predictor::moshi::executor::action
