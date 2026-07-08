#include "emel/model/qwen3/detail.hpp"

#include "emel/model/loader/errors.hpp"

namespace emel::model::qwen3::detail {

namespace {

constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "output_norm.weight";
constexpr uint32_t k_global_tensor_count = 3u;
constexpr uint32_t k_block_tensor_count = 10u;

}  // namespace

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  int32_t key_length = 0;
  int32_t value_length = 0;
  if (!loader.assign_i32("qwen3.context_length", model_out.params.n_ctx) ||
      !loader.assign_i32("qwen3.embedding_length", model_out.params.n_embd) ||
      !loader.assign_i32("qwen3.feed_forward_length", model_out.params.n_ff) ||
      !loader.assign_i32("qwen3.attention.head_count", model_out.params.n_head) ||
      !loader.assign_i32("qwen3.attention.head_count_kv", model_out.params.n_head_kv) ||
      !loader.assign_i32("qwen3.attention.key_length", key_length) ||
      !loader.assign_i32("qwen3.attention.value_length", value_length) ||
      !loader.assign_i32("qwen3.block_count", model_out.params.n_layer) ||
      !loader.assign_f32(
          "qwen3.attention.layer_norm_rms_epsilon", model_out.params.attention_layer_norm_rms_epsilon) ||
      !loader.assign_f32("qwen3.rope.freq_base", model_out.params.rope_freq_base)) {
    return false;
  }

  model_out.params.attention_key_length = key_length;
  model_out.params.attention_value_length = value_length;
  model_out.params.n_embd_out = model_out.params.n_embd;
  model_out.params.tie_word_embeddings = true;
  model_out.params.rope_pair_x0_stride = 1;
  model_out.params.rope_pair_x1_stride = 1;
  model_out.params.rope_pair_x1_offset = 0;
  model_out.params.rope_pair_x1_half_rot_offset = 1;
  if (model_out.params.n_rot == 0) {
    model_out.params.n_rot = key_length;
  }

  return key_length > 0 && value_length > 0;
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  emel::model::generation::contract contract = {};
  return build_generation_contract(model_data, contract);
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
  for (int32_t block_index = 0; block_index < execution.block_count;
       ++block_index) {
    auto &block = execution.blocks[static_cast<size_t>(block_index)];
    const auto block_err = emel::model::generation::bind_attention_block(
        model_data, block_index, true, false, block);
    if (block_err != emel::error::cast(emel::model::loader::error::none)) {
      contract_out.reset();
      return block_err;
    }
    auto &layer = descriptor.layers[static_cast<size_t>(block_index)];
    layer.residual_route =
        emel::model::generation::generation_residual_route::attention;
    layer.qk_norm_route = emel::model::generation::
        generation_attention_qk_norm_route::headwise_rms;
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
  }

  graph.execution = &execution;
  graph.tensor_count =
      k_global_tensor_count +
      static_cast<uint32_t>(execution.block_count) * k_block_tensor_count;
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

}  // namespace emel::model::qwen3::detail
