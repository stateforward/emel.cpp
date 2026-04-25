#pragma once

#include <algorithm>
#include <span>

#include "emel/diarization/sortformer/cache/detail.hpp"
#include "emel/diarization/sortformer/executor/context.hpp"
#include "emel/diarization/sortformer/executor/detail.hpp"
#include "emel/diarization/sortformer/executor/events.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"

namespace emel::diarization::sortformer::executor::action {

template <int32_t layer_index>
void execute_transformer_layer(context & ctx) noexcept {
  namespace transformer_detail = emel::diarization::sortformer::transformer::detail;
  const auto & layer_view = ctx.transformer.layers[static_cast<size_t>(layer_index)];
  constexpr size_t attention_cache_base = static_cast<size_t>(layer_index) * 4u;
  constexpr size_t feed_forward_cache_base = static_cast<size_t>(layer_index) * 2u;
  const auto & query_cache =
      ctx.transformer_workspace.attention_weight_caches[attention_cache_base + 0u];
  const auto & key_cache =
      ctx.transformer_workspace.attention_weight_caches[attention_cache_base + 1u];
  const auto & value_cache =
      ctx.transformer_workspace.attention_weight_caches[attention_cache_base + 2u];
  const auto & output_cache =
      ctx.transformer_workspace.attention_weight_caches[attention_cache_base + 3u];
  const auto & feed_forward_in_cache =
      ctx.transformer_workspace.feed_forward_weight_caches[feed_forward_cache_base + 0u];
  const auto & feed_forward_out_cache =
      ctx.transformer_workspace.feed_forward_weight_caches[feed_forward_cache_base + 1u];

  if constexpr ((layer_index % 2) == 0) {
    const std::span<const float> layer_input{ctx.hidden_a};
    std::span<float> layer_output{ctx.hidden_b};
    transformer_detail::compute_transformer_layer(
        layer_input,
        static_cast<uint32_t>(detail::k_frame_count),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.query_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.query_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.key_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.key_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.value_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.value_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.output_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.output_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_1_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_1_bias.tensor),
        detail::tensor_data<transformer_detail::k_inner_dim * transformer_detail::k_hidden_dim>(
            *layer_view.feed_forward_in_weight.tensor),
        detail::tensor_data<transformer_detail::k_inner_dim>(*layer_view.feed_forward_in_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_inner_dim>(
            *layer_view.feed_forward_out_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.feed_forward_out_bias.tensor),
        query_cache,
        key_cache,
        value_cache,
        output_cache,
        feed_forward_in_cache,
        feed_forward_out_cache,
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_2_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_2_bias.tensor),
        ctx.transformer_workspace,
        layer_output);
  } else {
    const std::span<const float> layer_input{ctx.hidden_b};
    std::span<float> layer_output{ctx.hidden_a};
    transformer_detail::compute_transformer_layer(
        layer_input,
        static_cast<uint32_t>(detail::k_frame_count),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.query_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.query_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.key_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.key_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.value_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.value_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>(
            *layer_view.output_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.output_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_1_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_1_bias.tensor),
        detail::tensor_data<transformer_detail::k_inner_dim * transformer_detail::k_hidden_dim>(
            *layer_view.feed_forward_in_weight.tensor),
        detail::tensor_data<transformer_detail::k_inner_dim>(*layer_view.feed_forward_in_bias.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim * transformer_detail::k_inner_dim>(
            *layer_view.feed_forward_out_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.feed_forward_out_bias.tensor),
        query_cache,
        key_cache,
        value_cache,
        output_cache,
        feed_forward_in_cache,
        feed_forward_out_cache,
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_2_weight.tensor),
        detail::tensor_data<transformer_detail::k_hidden_dim>(*layer_view.layer_norm_2_bias.tensor),
        ctx.transformer_workspace,
        layer_output);
  }

  if constexpr (layer_index + 1 < emel::diarization::sortformer::transformer::detail::k_layer_count) {
    execute_transformer_layer<layer_index + 1>(ctx);
  }
}

inline std::span<const float> final_hidden_frames(const context & ctx) noexcept {
  if constexpr ((emel::diarization::sortformer::transformer::detail::k_layer_count % 2) == 0) {
    return std::span<const float>{ctx.hidden_a};
  } else {
    return std::span<const float>{ctx.hidden_b};
  }
}

struct effect_begin_execute {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.request.frame_count_out = 0;
    runtime_ev.request.hidden_dim_out = 0;
  }
};

struct effect_mark_model_invalid {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::model_invalid);
  }
};

struct effect_mark_tensor_contract_invalid {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::tensor_contract);
  }
};

struct effect_mark_input_shape_invalid {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::input_shape);
  }
};

struct effect_mark_output_capacity_invalid {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::output_capacity);
  }
};

struct effect_bind_contracts {
  void operator()(const event::execute_run & runtime_ev, context & ctx) const noexcept {
    emel::diarization::sortformer::modules::detail::bind_contract(
        *runtime_ev.request.contract.model, ctx.modules);
    emel::diarization::sortformer::transformer::detail::bind_contract(
        *runtime_ev.request.contract.model, ctx.transformer);
    (void) emel::diarization::sortformer::modules::detail::
        prepare_encoder_projection_weight_cache(
            detail::tensor_data<emel::diarization::sortformer::modules::detail::k_hidden_dim *
                                emel::diarization::sortformer::modules::detail::k_encoder_dim>(
                *ctx.modules.encoder_projection_weight.tensor),
            ctx.encoder_projection_weight_cache);
    (void) emel::diarization::sortformer::transformer::detail::prepare_weight_caches(
        ctx.transformer,
        ctx.transformer_workspace);
  }
};

struct effect_execute_stage {
  void operator()(const event::execute_run & runtime_ev, context & ctx) const noexcept {
    namespace modules_detail = emel::diarization::sortformer::modules::detail;
    namespace transformer_detail = emel::diarization::sortformer::transformer::detail;

    emel::diarization::sortformer::cache::detail::reset(ctx.cache);
    std::fill(ctx.hidden_a.begin(), ctx.hidden_a.end(), 0.0f);
    std::fill(ctx.hidden_b.begin(), ctx.hidden_b.end(), 0.0f);

    const auto encoder_projection_weight =
        detail::tensor_data<modules_detail::k_hidden_dim * modules_detail::k_encoder_dim>(
            *ctx.modules.encoder_projection_weight.tensor);
    const auto encoder_projection_bias =
        detail::tensor_data<modules_detail::k_hidden_dim>(
            *ctx.modules.encoder_projection_bias.tensor);

    (void) modules_detail::compute_encoder_projection_batch(
        runtime_ev.request.encoder_frames,
        static_cast<size_t>(detail::k_frame_count),
        encoder_projection_weight,
        ctx.encoder_projection_weight_cache,
        encoder_projection_bias,
        ctx.transformer_workspace.dense_transposed_input,
        ctx.transformer_workspace.dense_transposed_output,
        ctx.hidden_a);

    for (int32_t frame = 0; frame < detail::k_frame_count; ++frame) {
      const size_t hidden_offset = static_cast<size_t>(frame) * detail::k_hidden_dim;
      auto hidden_frame = std::span<float, modules_detail::k_hidden_dim>{
          ctx.hidden_a.data() + hidden_offset,
          modules_detail::k_hidden_dim};
      emel::diarization::sortformer::cache::detail::write_frame(ctx.cache, frame, hidden_frame);
    }

    execute_transformer_layer<0>(ctx);

    const auto hidden_frames = final_hidden_frames(ctx);
    std::copy(hidden_frames.begin(),
              hidden_frames.end(),
              runtime_ev.request.hidden_out.begin());
    runtime_ev.request.frame_count_out = detail::k_frame_count;
    runtime_ev.request.hidden_dim_out = detail::k_hidden_dim;
    runtime_ev.ctx.err = detail::to_error(error::none);
  }
};

struct effect_store_success_error {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.request.on_done(events::execute_done{
      .request = &runtime_ev.request,
      .frame_count = runtime_ev.request.frame_count_out,
      .hidden_dim = runtime_ev.request.hidden_dim_out,
    });
  }
};

struct effect_publish_success_and_emit_done {
  void operator()(const event::execute_run & runtime_ev, context & ctx) const noexcept {
    effect_store_success_error{}(runtime_ev, ctx);
    effect_emit_done{}(runtime_ev, ctx);
  }
};

struct effect_store_error_error {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_error {
  void operator()(const event::execute_run & runtime_ev, context &) const noexcept {
    runtime_ev.request.on_error(events::execute_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_publish_error_and_emit_error {
  void operator()(const event::execute_run & runtime_ev, context & ctx) const noexcept {
    effect_store_error_error{}(runtime_ev, ctx);
    effect_emit_error{}(runtime_ev, ctx);
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_execute effect_begin_execute{};
inline constexpr effect_mark_model_invalid effect_mark_model_invalid{};
inline constexpr effect_mark_tensor_contract_invalid effect_mark_tensor_contract_invalid{};
inline constexpr effect_mark_input_shape_invalid effect_mark_input_shape_invalid{};
inline constexpr effect_mark_output_capacity_invalid effect_mark_output_capacity_invalid{};
inline constexpr effect_bind_contracts effect_bind_contracts{};
inline constexpr effect_execute_stage effect_execute_stage{};
inline constexpr effect_store_success_error effect_store_success_error{};
inline constexpr effect_emit_done effect_emit_done{};
inline constexpr effect_publish_success_and_emit_done effect_publish_success_and_emit_done{};
inline constexpr effect_store_error_error effect_store_error_error{};
inline constexpr effect_emit_error effect_emit_error{};
inline constexpr effect_publish_error_and_emit_error effect_publish_error_and_emit_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

}  // namespace emel::diarization::sortformer::executor::action
