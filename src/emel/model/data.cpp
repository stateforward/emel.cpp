#include "emel/model/data.hpp"

#include <array>
#include <cstdio>

#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model {

namespace {

constexpr std::string_view k_llama_architecture = "llama";
constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "output_norm.weight";
constexpr std::string_view k_output_name = "output.weight";

const data::tensor_record * find_tensor_by_name(const data & model_data,
                                                const std::string_view name) noexcept {
  for (uint32_t i = 0u; i < model_data.n_tensors; ++i) {
    const auto & tensor = model_data.tensors[i];
    if (tensor_name_view(model_data, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

bool tensor_has_storage(const data::tensor_record & tensor) noexcept {
  if (tensor.data == nullptr || tensor.data_size == 0u || tensor.n_dims <= 0) {
    return false;
  }

  for (int32_t i = 0; i < tensor.n_dims && i < static_cast<int32_t>(tensor.dims.size()); ++i) {
    if (tensor.dims[static_cast<size_t>(i)] <= 0) {
      return false;
    }
  }

  return true;
}

bool assign_tensor_view(const data & model_data,
                        const std::string_view name,
                        llama::detail::tensor_view & view_out) noexcept {
  const auto * tensor = find_tensor_by_name(model_data, name);
  if (tensor == nullptr || !tensor_has_storage(*tensor)) {
    return false;
  }

  view_out.tensor = tensor;
  view_out.name = tensor_name_view(model_data, *tensor);
  return true;
}

bool make_block_tensor_name(const int32_t block_index,
                            const std::string_view suffix,
                            std::array<char, 64> & buffer,
                            std::string_view & name_out) noexcept {
  const int written = std::snprintf(
      buffer.data(),
      buffer.size(),
      "blk.%d.%.*s",
      block_index,
      static_cast<int>(suffix.size()),
      suffix.data());
  if (written <= 0 || static_cast<size_t>(written) >= buffer.size()) {
    return false;
  }

  name_out = std::string_view{buffer.data(), static_cast<size_t>(written)};
  return true;
}

}  // namespace

std::string_view tensor_name_view(const data & model_data,
                                  const data::tensor_record & tensor) noexcept {
  const size_t begin = static_cast<size_t>(tensor.name_offset);
  const size_t length = static_cast<size_t>(tensor.name_length);
  if (begin + length > model_data.name_storage.size()) {
    return {};
  }

  return std::string_view{model_data.name_storage.data() + begin, length};
}

bool try_parse_block_index(const std::string_view name, int32_t & block_index_out) noexcept {
  constexpr std::string_view k_prefix = "blk.";
  if (!name.starts_with(k_prefix)) {
    return false;
  }

  size_t cursor = k_prefix.size();
  if (cursor >= name.size()) {
    return false;
  }

  int32_t value = 0;
  bool saw_digit = false;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    saw_digit = true;
    value = value * 10 + static_cast<int32_t>(name[cursor] - '0');
    ++cursor;
  }

  if (!saw_digit || cursor >= name.size() || name[cursor] != '.') {
    return false;
  }

  block_index_out = value;
  return true;
}

std::string_view architecture_name_view(const data & model_data) noexcept {
  size_t length = 0u;
  while (length < model_data.architecture_name.size() &&
         model_data.architecture_name[length] != '\0') {
    ++length;
  }

  return std::string_view{model_data.architecture_name.data(), length};
}

namespace llama::detail {

emel::error::type lookup_block_view(const execution_view & execution,
                                    const int32_t block_index,
                                    block_view & block_out) noexcept {
  if (execution.model == nullptr || block_index < 0 || block_index >= execution.block_count) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  block_out = {};
  block_out.index = block_index;

  std::array<char, 64> buffer = {};
  std::string_view name = {};
  const auto bind = [&](const std::string_view suffix, tensor_view & tensor_out) {
    return make_block_tensor_name(block_index, suffix, buffer, name) &&
           assign_tensor_view(*execution.model, name, tensor_out);
  };

  const bool ok = bind("attn_norm.weight", block_out.attention_norm) &&
                  bind("attn_q.weight", block_out.attention_q) &&
                  bind("attn_k.weight", block_out.attention_k) &&
                  bind("attn_v.weight", block_out.attention_v) &&
                  bind("attn_output.weight", block_out.attention_output) &&
                  bind("ffn_norm.weight", block_out.feed_forward_norm) &&
                  bind("ffn_gate.weight", block_out.feed_forward_gate) &&
                  bind("ffn_down.weight", block_out.feed_forward_down) &&
                  bind("ffn_up.weight", block_out.feed_forward_up);
  if (!ok) {
    block_out = {};
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type build_execution_view(const emel::model::data & model_data,
                                       execution_view & view_out) noexcept {
  view_out = {};

  if (architecture_name_view(model_data) != k_llama_architecture ||
      model_data.n_tensors == 0u ||
      model_data.n_layers <= 0 ||
      model_data.params.n_embd <= 0 ||
      model_data.params.n_ctx <= 0 ||
      model_data.weights_data == nullptr ||
      model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  execution_view view = {};
  view.model = &model_data;
  view.block_count = model_data.n_layers;

  if (!assign_tensor_view(model_data, k_token_embedding_name, view.token_embedding) ||
      !assign_tensor_view(model_data, k_output_norm_name, view.output_norm) ||
      !assign_tensor_view(model_data, k_output_name, view.output)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  for (int32_t i = 0; i < view.block_count; ++i) {
    block_view block = {};
    const auto err = lookup_block_view(view, i, block);
    if (err != emel::error::cast(emel::model::loader::error::none)) {
      return err;
    }
  }

  view_out = view;
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
      (static_cast<uint32_t>(execution.block_count) * k_block_tensor_count);
  topology_out.node_count = topology_out.tensor_count;
  topology_out.bytes_per_tensor = sizeof(float);
  topology_out.workspace_capacity_bytes =
      static_cast<uint64_t>(execution.model->params.n_embd) * sizeof(float) * 4u;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type build_step_plans(const topology & topology_in,
                                   step_plan & prefill_out,
                                   step_plan & decode_out) noexcept {
  prefill_out = {};
  decode_out = {};

  if (topology_in.execution == nullptr || topology_in.execution->model == nullptr ||
      topology_in.node_count == 0u || topology_in.tensor_count == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  prefill_out.graph = &topology_in;
  prefill_out.kind = step_kind::prefill;
  prefill_out.node_count = topology_in.node_count;
  prefill_out.tensor_count = topology_in.tensor_count;
  prefill_out.expected_outputs = 1;
  prefill_out.max_step_tokens = topology_in.execution->model->params.n_ctx;

  decode_out.graph = &topology_in;
  decode_out.kind = step_kind::decode;
  decode_out.node_count = topology_in.node_count;
  decode_out.tensor_count = topology_in.tensor_count;
  decode_out.expected_outputs = 1;
  decode_out.max_step_tokens = 1;

  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace llama::detail

}  // namespace emel::model
