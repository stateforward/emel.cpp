#pragma once

#include "emel/model/moshi/detail.hpp"
#include "emel/speech/generator/moshi/executor/context.hpp"
#include "emel/speech/generator/moshi/executor/detail.hpp"
#include "emel/speech/generator/moshi/executor/events.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace emel::speech::generator::moshi::executor::action {

namespace detail_ns {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail_ns

namespace sampling_ns {

inline int32_t bounded_top_k(const int32_t requested,
                             const int32_t card) noexcept {
  int32_t top_k = requested;
  if (top_k > card) {
    top_k = card;
  }
  if (top_k < 0) {
    top_k = 0;
  }
  return top_k;
}

inline void reset_top_k(event::step_ctx &ctx, const int32_t top_k) noexcept {
  for (int32_t slot = 0; slot < top_k; ++slot) {
    ctx.top_scores[static_cast<size_t>(slot)] =
        -std::numeric_limits<float>::infinity();
    ctx.top_indices[static_cast<size_t>(slot)] = -1;
  }
}

inline void insert_top_k(event::step_ctx &ctx, const int32_t top_k,
                         const int32_t index, const float score) noexcept {
  for (int32_t slot = 0; slot < top_k; ++slot) {
    if (score > ctx.top_scores[static_cast<size_t>(slot)]) {
      for (int32_t move = top_k - 1; move > slot; --move) {
        ctx.top_scores[static_cast<size_t>(move)] =
            ctx.top_scores[static_cast<size_t>(move - 1)];
        ctx.top_indices[static_cast<size_t>(move)] =
            ctx.top_indices[static_cast<size_t>(move - 1)];
      }
      ctx.top_scores[static_cast<size_t>(slot)] = score;
      ctx.top_indices[static_cast<size_t>(slot)] = index;
      return;
    }
  }
}

inline bool
consume_reference_random_draws(const int32_t card,
                               const int32_t requested_top_k) noexcept {
  const int32_t top_k = bounded_top_k(requested_top_k, card);
  if (card <= 0 || top_k <= 0 ||
      static_cast<uint64_t>(top_k) > detail::k_max_sampling_top_k) {
    return false;
  }
  for (int32_t index = 0; index < top_k; ++index) {
    (void)std::rand();
  }
  return true;
}

inline bool sample_top_k_exponential(event::step_ctx &ctx, const int32_t card,
                                     const int32_t requested_top_k,
                                     const float temperature) noexcept {
  const int32_t top_k = bounded_top_k(requested_top_k, card);
  if (card <= 0 || temperature <= 0.0f || top_k <= 0 ||
      static_cast<uint64_t>(card) > detail::k_max_sampling_card ||
      static_cast<uint64_t>(top_k) > detail::k_max_sampling_top_k) {
    ctx.best_index = -1;
    ctx.best_score = 0.0f;
    return false;
  }

  reset_top_k(ctx, top_k);
  for (int32_t index = 0; index < card; ++index) {
    insert_top_k(ctx, top_k, index, ctx.logits[static_cast<size_t>(index)]);
  }

  const float max_score = ctx.top_scores[0];
  float best_q = -std::numeric_limits<float>::infinity();
  int32_t best_index = -1;
  for (int32_t slot = 0; slot < top_k; ++slot) {
    const float uniform =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    const float exponential = -std::log(uniform);
    const float weight = std::exp(
        (ctx.top_scores[static_cast<size_t>(slot)] - max_score) / temperature);
    const float q = weight / exponential;
    if (q > best_q) {
      best_q = q;
      best_index = ctx.top_indices[static_cast<size_t>(slot)];
    }
  }

  ctx.best_index = best_index;
  ctx.best_score = best_q;
  return best_index >= 0;
}

} // namespace sampling_ns

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
    ctx.sampling.audio_top_k = runtime_ev.request.sampling_audio_top_k;
    ctx.sampling.text_top_k = runtime_ev.request.sampling_text_top_k;
    ctx.sampling.seed = runtime_ev.request.sampling_seed;
    std::srand(static_cast<unsigned int>(ctx.sampling.seed));
    (void)emel::model::moshi::detail::build_execution_contract(
        runtime_ev.request.model, ctx.session.contract);
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
    runtime_ev.ctx.text_sampling_ok = false;
    runtime_ev.ctx.depformer_kv_bound = false;
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
    runtime_ev.ctx.depformer_sampling_ok = false;
    runtime_ev.ctx.best_index = -1;
    runtime_ev.ctx.best_score = 0.0f;
    runtime_ev.ctx.input_audio_codebook_index = 0;
    runtime_ev.ctx.temporal_layer_index = 0;
    runtime_ev.ctx.temporal_logical_position = -1;
    runtime_ev.ctx.temporal_physical_position = -1;
    runtime_ev.ctx.depformer_codebook_index = 0;
    runtime_ev.ctx.depformer_weight_index = -1;
    runtime_ev.ctx.depformer_layer_index = 0;
    runtime_ev.ctx.depformer_logical_position = -1;
    runtime_ev.ctx.depformer_physical_position = -1;
    runtime_ev.ctx.depformer_valid_positions = 0;
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
    detail::set_op_param_f32(rms_ev, 0u, detail::k_rms_norm_eps);
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

struct effect_apply_temporal_layer_rope {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t head_dim = hidden_dim / lm.num_heads;
    const int32_t half = head_dim / 2;
    runtime_ev.ctx.temporal_layer_rope_ok = false;
    runtime_ev.ctx.temporal_logical_position = detail::current_logical_position(
        runtime_ev.request.memory_snapshot, runtime_ev.request.sequence_id);
    runtime_ev.ctx.temporal_physical_position = detail::physical_position(
        runtime_ev.request.memory_snapshot, runtime_ev.request.sequence_id,
        runtime_ev.ctx.temporal_logical_position);

    const float neg_log_period = -std::log(static_cast<float>(lm.max_period));
    const float position =
        static_cast<float>(runtime_ev.ctx.temporal_logical_position);
    for (int32_t part = 0; part < 2; ++part) {
      float *data = runtime_ev.ctx.qkv.data() +
                    static_cast<size_t>(part) * static_cast<size_t>(hidden_dim);
      for (int32_t head = 0; head < lm.num_heads; ++head) {
        float *head_data =
            data + static_cast<size_t>(head) * static_cast<size_t>(head_dim);
        for (int32_t index = 0; index < half; ++index) {
          const float freq =
              std::exp(neg_log_period * static_cast<float>(index) /
                       static_cast<float>(half));
          const float arg = position * freq;
          const float rotr = std::cos(arg);
          const float roti = std::sin(arg);
          const float real = head_data[static_cast<size_t>(2 * index)];
          const float imag = head_data[static_cast<size_t>(2 * index + 1)];
          runtime_ev.ctx.rope[static_cast<size_t>(index)] =
              real * rotr - imag * roti;
          runtime_ev.ctx.rope[static_cast<size_t>(half + index)] =
              real * roti + imag * rotr;
        }
        std::memcpy(head_data, runtime_ev.ctx.rope.data(),
                    static_cast<size_t>(head_dim) * sizeof(float));
      }
    }
    runtime_ev.ctx.temporal_layer_rope_ok = true;
  }
};

struct effect_write_temporal_layer_kv_cache {
  void operator()(const event::step_run &runtime_ev,
                  const context &ctx) const noexcept {
    const int32_t hidden_dim = ctx.session.hidden_dim;
    runtime_ev.ctx.temporal_layer_cache_write_ok = false;
    auto &view = runtime_ev.ctx.temporal_kv;
    const size_t layer_offset = view.layer_cache_offsets[static_cast<size_t>(
        runtime_ev.ctx.temporal_layer_index)];
    const size_t physical_position =
        static_cast<size_t>(runtime_ev.ctx.temporal_physical_position);
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
    const auto &snapshot = runtime_ev.request.memory_snapshot;
    const auto &view = runtime_ev.ctx.temporal_kv;
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t hidden_dim = ctx.session.hidden_dim;
    const int32_t head_dim = hidden_dim / lm.num_heads;
    const int32_t capacity = view.position_capacity;
    const int32_t sequence_length =
        snapshot.sequence_length(runtime_ev.request.sequence_id);
    const int32_t current_logical = runtime_ev.ctx.temporal_logical_position;
    const int32_t logical_begin =
        sequence_length > capacity ? sequence_length - capacity : 0;
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
      for (int32_t logical = logical_begin; logical < sequence_length;
           ++logical) {
        const int32_t physical = detail::physical_position(
            snapshot, runtime_ev.request.sequence_id, logical);
        const int32_t delta = current_logical - logical;
        const bool visible =
            delta >= 0 && (lm.context <= 0 || delta < lm.context);
        const size_t cache_begin =
            layer_offset +
            static_cast<size_t>(physical) * static_cast<size_t>(hidden_dim) +
            static_cast<size_t>(head_offset);
        runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)] =
            visible ? emel::kernel::detail::vec_dot_bf16_ggml(
                          head_dim, view.key_cache.data() + cache_begin,
                          runtime_ev.ctx.q_bf16.data()) *
                          scale
                    : -std::numeric_limits<float>::infinity();
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
        for (int32_t logical = logical_begin; logical < sequence_length;
             ++logical) {
          const int32_t physical = detail::physical_position(
              snapshot, runtime_ev.request.sequence_id, logical);
          const size_t value_index =
              layer_offset +
              static_cast<size_t>(physical) * static_cast<size_t>(hidden_dim) +
              static_cast<size_t>(head_offset + dim);
          sum +=
              static_cast<double>(
                  emel::kernel::detail::bf16_to_fp32(
                      view.value_cache[value_index]) *
                  emel::kernel::detail::bf16_to_fp32(
                      runtime_ev.ctx.attention_weights_bf16[static_cast<size_t>(
                          physical)]));
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
    detail::set_op_param_f32(rms_ev, 0u, detail::k_rms_norm_eps);
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
    runtime_ev.ctx.temporal_logical_position = -1;
    runtime_ev.ctx.temporal_physical_position = -1;
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
    detail::set_op_param_f32(rms_ev, 0u, detail::k_rms_norm_eps);
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

struct effect_sample_text_token {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.text_sampling_ok = sampling_ns::sample_top_k_exponential(
        runtime_ev.ctx, ctx.session.text_card, ctx.sampling.text_top_k,
        ctx.sampling.text_temperature);
    runtime_ev.ctx.text_logits_ok = runtime_ev.ctx.text_sampling_ok;
    runtime_ev.request.text_token_out = runtime_ev.ctx.best_index;
  }
};

struct effect_consume_forced_text_sampling {
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
    runtime_ev.ctx.text_sampling_ok = sampling_ns::sample_top_k_exponential(
        runtime_ev.ctx, ctx.session.text_card, ctx.sampling.text_top_k,
        ctx.sampling.text_temperature);
    runtime_ev.ctx.text_logits_ok =
        runtime_ev.ctx.text_logits_ok && runtime_ev.ctx.text_sampling_ok;
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
    runtime_ev.ctx.depformer_kv = {};
    runtime_ev.ctx.depformer_kv_bound = ctx.depformer_kv.bind(
        ctx.depformer_kv.cache, runtime_ev.request.model,
        runtime_ev.request.memory_snapshot, runtime_ev.request.sequence_id,
        runtime_ev.ctx.depformer_kv);
    if (runtime_ev.ctx.depformer_kv_bound &&
        runtime_ev.ctx.depformer_kv.offset != nullptr) {
      *runtime_ev.ctx.depformer_kv.offset = 0;
    }
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
    detail::set_op_param_f32(rms_ev, 0u, detail::k_rms_norm_eps);
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

    const int32_t logical_position = *view.offset;
    const int32_t physical_position = logical_position % view.position_capacity;
    runtime_ev.ctx.depformer_logical_position = logical_position;
    runtime_ev.ctx.depformer_physical_position = physical_position;
    runtime_ev.ctx.depformer_valid_positions =
        std::min(logical_position + 1, view.position_capacity);

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
    const auto &lm = runtime_ev.request.model.moshi_lm;
    const int32_t dep_dim = lm.depformer_dim;
    const int32_t head_dim = dep_dim / lm.depformer_num_heads;
    const int32_t capacity = view.position_capacity;
    const int32_t logical_position = runtime_ev.ctx.depformer_logical_position;
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
        const bool visible =
            logical_position >= capacity || physical <= logical_position;
        const size_t cache_begin =
            layer_offset +
            static_cast<size_t>(physical) * static_cast<size_t>(dep_dim) +
            static_cast<size_t>(head_offset);
        runtime_ev.ctx.attention_scores[static_cast<size_t>(physical)] =
            visible ? emel::kernel::detail::vec_dot_bf16_ggml(
                          head_dim, view.key_cache.data() + cache_begin,
                          runtime_ev.ctx.q_bf16.data()) *
                          scale
                    : -std::numeric_limits<float>::infinity();
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
        for (int32_t physical = 0; physical < capacity; ++physical) {
          const size_t value_index =
              layer_offset +
              static_cast<size_t>(physical) * static_cast<size_t>(dep_dim) +
              static_cast<size_t>(head_offset + dim);
          sum +=
              static_cast<double>(
                  emel::kernel::detail::bf16_to_fp32(
                      view.value_cache[value_index]) *
                  emel::kernel::detail::bf16_to_fp32(
                      runtime_ev.ctx.attention_weights_bf16[static_cast<size_t>(
                          physical)]));
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
    detail::set_op_param_f32(rms_ev, 0u, detail::k_rms_norm_eps);
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
        "gating.%d.linear_in.weight",
        runtime_ev.ctx.depformer_weight_index);
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
        "gating.%d.linear_in.weight",
        runtime_ev.ctx.depformer_weight_index);
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
    runtime_ev.ctx.depformer_logical_position = -1;
    runtime_ev.ctx.depformer_physical_position = -1;
    runtime_ev.ctx.depformer_valid_positions = 0;
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

struct effect_sample_depformer_token {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.depformer_sampling_ok =
        sampling_ns::sample_top_k_exponential(
            runtime_ev.ctx, ctx.session.audio_card, ctx.sampling.audio_top_k,
            ctx.sampling.audio_temperature);
    runtime_ev.ctx.depformer_logits_ok = runtime_ev.ctx.depformer_sampling_ok;
  }
};

struct effect_publish_depformer_token {
  void operator()(const event::step_run &runtime_ev, context &) const noexcept {
    const int32_t codebook = runtime_ev.ctx.depformer_codebook_index;
    runtime_ev.request.audio_tokens_out[static_cast<size_t>(codebook)] =
        runtime_ev.ctx.best_index;
    ++(*runtime_ev.ctx.depformer_kv.offset);
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
    runtime_ev.ctx.depformer_logical_position = -1;
    runtime_ev.ctx.depformer_physical_position = -1;
    runtime_ev.ctx.depformer_valid_positions = 0;
    runtime_ev.ctx.depformer_weight_index = -1;
  }
};

struct effect_bind_temporal_kv {
  void operator()(const event::step_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.temporal_kv = {};
    runtime_ev.ctx.temporal_kv_bound = ctx.temporal_kv.bind(
        ctx.temporal_kv.cache, runtime_ev.request.model,
        runtime_ev.request.memory_snapshot, runtime_ev.request.sequence_id,
        runtime_ev.ctx.temporal_kv);
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

} // namespace emel::speech::generator::moshi::executor::action
