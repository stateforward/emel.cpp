#include "emel/model/lfm2/detail.hpp"

#include <array>
#include <span>

#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::lfm2::detail {

namespace {

constexpr std::string_view k_architecture = "lfm2";
constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "token_embd_norm.weight";
constexpr uint32_t k_global_tensor_count = 3u;
constexpr uint32_t k_attention_block_tensor_count = 11u;
constexpr uint32_t k_conv_block_tensor_count = 8u;
// Maintained LFM2.5-1.2B geometry.
constexpr int32_t k_block_count = 16;
constexpr int32_t k_context_length = 128000;
constexpr int32_t k_embedding_length = 2048;
constexpr int32_t k_head_count = 32;
constexpr int32_t k_head_count_kv = 8;
constexpr int32_t k_vocab_size = 65536;
constexpr int32_t k_shortconv_l_cache = 3;
constexpr float k_rope_freq_base = 1000000.0f;
// Maintained LFM2.5-230M geometry (shares vocab, l_cache, ctx, and freq base).
constexpr int32_t k_230m_block_count = 14;
constexpr int32_t k_230m_embedding_length = 1024;
constexpr int32_t k_230m_head_count = 16;
// 1.2B layout fallback for model data without a bound per-layer kv-head
// pattern (synthetic fixtures); real models classify from
// `lfm2.attention.head_count_kv`, where a nonzero entry marks attention.
constexpr std::array<int32_t, 6> k_attention_layers = {2, 5, 8, 10, 12, 14};

bool is_attention_layer(const emel::model::data & model_data,
                        const int32_t block_index) noexcept {
  const uint32_t pattern_count = model_data.params.attention_layer_pattern_count;
  if (pattern_count > 0u) {
    return block_index >= 0 && static_cast<uint32_t>(block_index) < pattern_count &&
           model_data.params
                   .attention_layer_pattern_flags[static_cast<size_t>(block_index)] != 0u;
  }
  for (const int32_t candidate : k_attention_layers) {
    if (candidate == block_index) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool is_execution_architecture(const std::string_view architecture) noexcept {
  return architecture == k_architecture;
}

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  if (!loader.assign_i32("lfm2.context_length", model_out.params.n_ctx) ||
      !loader.assign_i32("lfm2.embedding_length", model_out.params.n_embd) ||
      !loader.assign_i32("lfm2.feed_forward_length", model_out.params.n_ff) ||
      !loader.assign_i32("lfm2.attention.head_count", model_out.params.n_head) ||
      !loader.assign_i32("lfm2.block_count", model_out.params.n_layer) ||
      !loader.assign_i32("lfm2.vocab_size", model_out.params.n_vocab) ||
      !loader.assign_i32("lfm2.shortconv.l_cache", model_out.params.shortconv_l_cache) ||
      !loader.assign_f32(
          "lfm2.attention.layer_norm_rms_epsilon", model_out.params.attention_layer_norm_rms_epsilon) ||
      !loader.assign_f32("lfm2.rope.freq_base", model_out.params.rope_freq_base) ||
      !loader.assign_first_nonzero_i32_from_array(
          "lfm2.attention.head_count_kv", model_out.params.n_head_kv) ||
      !loader.copy_flag_array(
          "lfm2.attention.head_count_kv",
          std::span<uint8_t>{model_out.params.attention_layer_pattern_flags},
          model_out.params.attention_layer_pattern_count)) {
    return false;
  }

  model_out.params.n_embd_out = model_out.params.n_embd;
  model_out.params.tie_word_embeddings = true;
  model_out.params.rope_pair_x0_stride = 1;
  model_out.params.rope_pair_x1_stride = 1;
  model_out.params.rope_pair_x1_offset = 0;
  model_out.params.rope_pair_x1_half_rot_offset = 1;
  return true;
}

namespace {

emel::error::type validate_contract(const emel::model::data & model_data,
                                    const bool strict_metadata) noexcept {
  const bool metadata_1_2b =
      model_data.n_layers == k_block_count &&
      model_data.params.n_layer == k_block_count &&
      model_data.params.n_embd == k_embedding_length &&
      model_data.params.n_head == k_head_count;
  const bool metadata_230m =
      model_data.n_layers == k_230m_block_count &&
      model_data.params.n_layer == k_230m_block_count &&
      model_data.params.n_embd == k_230m_embedding_length &&
      model_data.params.n_head == k_230m_head_count;
  const bool metadata_ok = strict_metadata
      ? (metadata_1_2b || metadata_230m) &&
            model_data.params.n_ctx == k_context_length &&
            model_data.params.n_head_kv == k_head_count_kv &&
            model_data.params.n_vocab == k_vocab_size &&
            model_data.params.shortconv_l_cache == k_shortconv_l_cache &&
            model_data.params.rope_freq_base == k_rope_freq_base
      : emel::model::architecture_name_view(model_data) == "lfm2" &&
            model_data.n_layers > 0 &&
            model_data.params.n_layer == model_data.n_layers &&
            model_data.params.n_ctx > 0 &&
            model_data.params.n_embd > 0 &&
            model_data.params.n_head > 0 &&
            model_data.params.n_head_kv > 0 &&
            model_data.params.n_vocab > 0 &&
            model_data.params.shortconv_l_cache > 0 &&
            model_data.params.rope_freq_base > 0.0f;
  if (!metadata_ok ||
      !emel::model::generation::has_tensor_named(model_data, k_token_embedding_name) ||
      !emel::model::generation::has_tensor_named(model_data, k_output_norm_name)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  for (int32_t block_index = 0; block_index < model_data.n_layers; ++block_index) {
    const bool common_ok =
        emel::model::generation::require_block_tensor(
            model_data, block_index, "attn_norm.weight") &&
        emel::model::generation::require_block_tensor(
            model_data, block_index, "ffn_norm.weight") &&
        emel::model::generation::require_block_tensor(
            model_data, block_index, "ffn_gate.weight") &&
        emel::model::generation::require_block_tensor(
            model_data, block_index, "ffn_down.weight") &&
        emel::model::generation::require_block_tensor(
            model_data, block_index, "ffn_up.weight");
    if (!common_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    const bool attention_layer = is_attention_layer(model_data, block_index);
    const bool hybrid_ok =
        attention_layer
            ? emel::model::generation::require_block_tensor(
                  model_data, block_index, "attn_q.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "attn_k.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "attn_v.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "attn_q_norm.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "attn_k_norm.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "attn_output.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "shortconv.conv.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "shortconv.in_proj.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "shortconv.out_proj.weight")
            : emel::model::generation::require_block_tensor(
                  model_data, block_index, "shortconv.conv.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "shortconv.in_proj.weight") &&
                  emel::model::generation::require_block_tensor(
                      model_data, block_index, "shortconv.out_proj.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "attn_q.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "attn_k.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "attn_v.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "attn_q_norm.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "attn_k_norm.weight") &&
                  emel::model::generation::reject_block_tensor(
                      model_data, block_index, "attn_output.weight");
    if (!hybrid_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace

emel::error::type validate_builder_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, false);
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  return validate_builder_contract(model_data);
}

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, true);
}

emel::error::type build_generation_contract(
    const emel::model::data & model_data,
    emel::model::generation::contract & contract_out) noexcept {
  contract_out.reset();
  auto &execution = contract_out.execution;
  auto &descriptor = contract_out.generation_execution;
  auto &graph = contract_out.topology;

  if (model_data.n_tensors == 0u || model_data.n_layers <= 0 ||
      model_data.params.n_embd <= 0 || model_data.params.n_ctx <= 0 ||
      static_cast<uint32_t>(model_data.n_layers) >
          emel::model::generation::execution_view::k_max_blocks) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  execution.model = &model_data;
  execution.block_count = model_data.n_layers;
  execution.blocks.resize(static_cast<size_t>(execution.block_count));
  if (!emel::model::generation::bind_tensor_view(
          model_data, k_token_embedding_name, execution.token_embedding) ||
      !emel::model::generation::bind_tensor_view(
          model_data, k_output_norm_name, execution.output_norm) ||
      !emel::model::generation::bind_output_view(
          model_data, execution.token_embedding, true, execution.output)) {
    contract_out.reset();
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  descriptor.execution = &execution;
  descriptor.layer_count = static_cast<uint32_t>(execution.block_count);
  uint32_t tensor_count = k_global_tensor_count;
  for (int32_t block_index = 0; block_index < execution.block_count;
       ++block_index) {
    auto &block = execution.blocks[static_cast<size_t>(block_index)];
    const bool attention_layer = is_attention_layer(model_data, block_index);
    const auto block_err =
        attention_layer
            ? emel::model::generation::bind_attention_block(
                  model_data, block_index, true, false, block)
            : emel::model::generation::bind_shortconv_block(
                  model_data, block_index, block);
    if (block_err != emel::error::cast(emel::model::loader::error::none)) {
      contract_out.reset();
      return block_err;
    }
    auto &layer = descriptor.layers[static_cast<size_t>(block_index)];
    layer.residual_route =
        attention_layer
            ? emel::model::generation::generation_residual_route::attention
            : emel::model::generation::generation_residual_route::shortconv;
    layer.qk_norm_route =
        attention_layer
            ? emel::model::generation::generation_attention_qk_norm_route::
                  headwise_rms
            : emel::model::generation::generation_attention_qk_norm_route::none;
    layer.value_route = emel::model::generation::
        generation_attention_value_route::dedicated_value;
    layer.v_norm_route =
        emel::model::generation::generation_attention_v_norm_route::none;
    layer.window_route =
        emel::model::generation::generation_attention_window_route::full_context;
    layer.attention_key_length = model_data.params.attention_key_length;
    layer.attention_value_length = model_data.params.attention_value_length;
    layer.attention_rope_dim = model_data.params.n_rot;
    layer.attention_rope_freq_base = model_data.params.rope_freq_base;
    tensor_count += attention_layer ? k_attention_block_tensor_count
                                    : k_conv_block_tensor_count;
  }

  graph.execution = &execution;
  graph.tensor_count = tensor_count;
  graph.node_count = graph.tensor_count;
  graph.bytes_per_tensor = sizeof(float);
  graph.workspace_capacity_bytes =
      static_cast<uint64_t>(graph.tensor_count) *
      static_cast<uint64_t>(model_data.params.n_embd) * sizeof(float);

  const auto complete_err = emel::model::generation::complete_contract(contract_out);
  if (complete_err != emel::error::cast(emel::model::loader::error::none)) {
    contract_out.reset();
    return complete_err;
  }

  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace emel::model::lfm2::detail
