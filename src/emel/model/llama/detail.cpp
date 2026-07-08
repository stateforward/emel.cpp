#include "emel/model/llama/detail.hpp"

#include "emel/model/loader/errors.hpp"

namespace emel::model::llama::detail {

namespace {

constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "output_norm.weight";

}  // namespace

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  const bool loaded =
      loader.assign_i32("llama.context_length", model_out.params.n_ctx) &&
      loader.assign_i32("llama.embedding_length", model_out.params.n_embd) &&
      loader.assign_i32("llama.embedding_length_out", model_out.params.n_embd_out) &&
      loader.assign_i32("llama.feed_forward_length", model_out.params.n_ff) &&
      loader.assign_i32("llama.attention.head_count", model_out.params.n_head) &&
      loader.assign_i32("llama.attention.head_count_kv", model_out.params.n_head_kv) &&
      loader.assign_i32("llama.rope.dimension_count", model_out.params.n_rot) &&
      loader.assign_i32("llama.block_count", model_out.params.n_layer) &&
      loader.assign_i32("llama.vocab_size", model_out.params.n_vocab) &&
      loader.assign_f32(
          "llama.attention.layer_norm_epsilon", model_out.params.attention_layer_norm_epsilon) &&
      loader.assign_f32(
          "llama.attention.layer_norm_rms_epsilon",
          model_out.params.attention_layer_norm_rms_epsilon) &&
      loader.assign_f32("llama.attention.clamp_kqv", model_out.params.attention_clamp_kqv) &&
      loader.assign_f32("llama.attn_logit_softcapping", model_out.params.attn_logit_softcapping) &&
      loader.assign_f32(
          "llama.final_logit_softcapping", model_out.params.final_logit_softcapping) &&
      loader.assign_f32("llama.residual_scale", model_out.params.residual_scale) &&
      loader.assign_f32("llama.embedding_scale", model_out.params.embedding_scale) &&
      loader.assign_f32("llama.rope.freq_base", model_out.params.rope_freq_base) &&
      loader.assign_f32("llama.rope.freq_base_swa", model_out.params.rope_freq_base_swa);
  return loaded;
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  execution_view view = {};
  return build_execution_view(model_data, view);
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
          model_data, execution.token_embedding, false, execution.output)) {
    contract_out.reset();
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  descriptor.execution = &execution;
  descriptor.layer_count = static_cast<uint32_t>(execution.block_count);
  for (int32_t block_index = 0; block_index < execution.block_count;
       ++block_index) {
    auto &block = execution.blocks[static_cast<size_t>(block_index)];
    const auto block_err = emel::model::generation::bind_attention_block(
        model_data, block_index, false, false, block);
    if (block_err != emel::error::cast(emel::model::loader::error::none)) {
      contract_out.reset();
      return block_err;
    }
    auto &layer = descriptor.layers[static_cast<size_t>(block_index)];
    layer.residual_route = generation_residual_route::attention;
    layer.qk_norm_route = generation_attention_qk_norm_route::none;
    layer.value_route = generation_attention_value_route::dedicated_value;
    layer.v_norm_route = generation_attention_v_norm_route::none;
    layer.window_route = generation_attention_window_route::full_context;
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

emel::error::type build_execution_view(const emel::model::data & model_data,
                                       execution_view & view_out) noexcept {
  emel::model::generation::contract contract{};
  const auto err = build_generation_contract(model_data, contract);
  if (err != emel::error::cast(emel::model::loader::error::none)) {
    view_out = {};
    return err;
  }
  view_out = contract.execution;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type build_generation_execution_descriptor(
    const execution_view & execution,
    generation_execution_descriptor & descriptor_out) noexcept {
  descriptor_out = {};
  if (execution.model == nullptr || execution.block_count <= 0 ||
      static_cast<uint32_t>(execution.block_count) >
          generation_execution_descriptor::k_max_layers) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  descriptor_out.execution = &execution;
  descriptor_out.layer_count = static_cast<uint32_t>(execution.block_count);
  for (int32_t block_index = 0; block_index < execution.block_count;
       ++block_index) {
    auto &layer = descriptor_out.layers[static_cast<size_t>(block_index)];
    layer.residual_route = generation_residual_route::attention;
    layer.qk_norm_route = generation_attention_qk_norm_route::none;
    layer.value_route = generation_attention_value_route::dedicated_value;
    layer.v_norm_route = generation_attention_v_norm_route::none;
    layer.window_route = generation_attention_window_route::full_context;
    layer.attention_key_length = execution.model->params.attention_key_length;
    layer.attention_value_length = execution.model->params.attention_value_length;
    layer.attention_rope_dim = execution.model->params.n_rot;
    layer.attention_rope_freq_base = execution.model->params.rope_freq_base;
  }

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type build_topology(const execution_view & execution,
                                 topology & topology_out) noexcept {
  topology_out = {};
  if (execution.model == nullptr || execution.block_count <= 0 ||
      execution.model->params.n_embd <= 0) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  topology_out.execution = &execution;
  topology_out.tensor_count =
      k_global_tensor_count +
      static_cast<uint32_t>(execution.block_count) * k_block_tensor_count;
  topology_out.node_count = topology_out.tensor_count;
  topology_out.bytes_per_tensor = sizeof(float);
  topology_out.workspace_capacity_bytes =
      static_cast<uint64_t>(topology_out.tensor_count) *
      static_cast<uint64_t>(execution.model->params.n_embd) * sizeof(float);
  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace emel::model::llama::detail
