#include "emel/model/data.hpp"

#include <array>
#include <cstdio>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
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

constexpr std::array<llama::detail::quantized_stage_family,
                     llama::detail::k_quantized_stage_family_count>
    k_stage_families = {
        llama::detail::quantized_stage_family::token_embedding,
        llama::detail::quantized_stage_family::output_norm,
        llama::detail::quantized_stage_family::output,
        llama::detail::quantized_stage_family::attention_norm,
        llama::detail::quantized_stage_family::attention_q,
        llama::detail::quantized_stage_family::attention_k,
        llama::detail::quantized_stage_family::attention_v,
        llama::detail::quantized_stage_family::attention_output,
        llama::detail::quantized_stage_family::feed_forward_norm,
        llama::detail::quantized_stage_family::feed_forward_gate,
        llama::detail::quantized_stage_family::feed_forward_down,
        llama::detail::quantized_stage_family::feed_forward_up,
    };

bool is_supported_quantized_type(const int32_t tensor_type) noexcept {
  return ::emel::kernel::detail::is_quantized_k_dtype(static_cast<uint8_t>(tensor_type));
}

bool is_f32_type(const int32_t tensor_type) noexcept {
  return static_cast<uint8_t>(tensor_type) == ::emel::kernel::detail::dtype_f32;
}

bool is_vector_dequant_stage(const llama::detail::quantized_stage_family family) noexcept {
  return family == llama::detail::quantized_stage_family::token_embedding ||
         family == llama::detail::quantized_stage_family::output_norm ||
         family == llama::detail::quantized_stage_family::attention_norm ||
         family == llama::detail::quantized_stage_family::feed_forward_norm;
}

llama::detail::quantized_contract_kind classify_contract(
    const llama::detail::quantized_stage_family family,
    const int32_t tensor_type,
    const bool consistent_across_layers) noexcept {
  if (!consistent_across_layers) {
    return llama::detail::quantized_contract_kind::explicit_no_claim;
  }

  if (is_vector_dequant_stage(family)) {
    return is_f32_type(tensor_type)
               ? llama::detail::quantized_contract_kind::approved_dense_f32_by_contract
               : is_supported_quantized_type(tensor_type)
                     ? llama::detail::quantized_contract_kind::approved_dense_f32_by_contract
                     : llama::detail::quantized_contract_kind::explicit_no_claim;
  }

  if (is_supported_quantized_type(tensor_type)) {
    return llama::detail::quantized_contract_kind::native_quantized;
  }

  return is_f32_type(tensor_type)
             ? llama::detail::quantized_contract_kind::approved_dense_f32_by_contract
               : llama::detail::quantized_contract_kind::explicit_no_claim;
}

const data::tensor_record * stage_tensor(const llama::detail::execution_view & execution,
                                         const llama::detail::block_view * block,
                                         const llama::detail::quantized_stage_family family) noexcept {
  switch (family) {
    case llama::detail::quantized_stage_family::token_embedding:
      return execution.token_embedding.tensor;
    case llama::detail::quantized_stage_family::output_norm:
      return execution.output_norm.tensor;
    case llama::detail::quantized_stage_family::output:
      return execution.output.tensor;
    case llama::detail::quantized_stage_family::attention_norm:
      return block != nullptr ? block->attention_norm.tensor : nullptr;
    case llama::detail::quantized_stage_family::attention_q:
      return block != nullptr ? block->attention_q.tensor : nullptr;
    case llama::detail::quantized_stage_family::attention_k:
      return block != nullptr ? block->attention_k.tensor : nullptr;
    case llama::detail::quantized_stage_family::attention_v:
      return block != nullptr ? block->attention_v.tensor : nullptr;
    case llama::detail::quantized_stage_family::attention_output:
      return block != nullptr ? block->attention_output.tensor : nullptr;
    case llama::detail::quantized_stage_family::feed_forward_norm:
      return block != nullptr ? block->feed_forward_norm.tensor : nullptr;
    case llama::detail::quantized_stage_family::feed_forward_gate:
      return block != nullptr ? block->feed_forward_gate.tensor : nullptr;
    case llama::detail::quantized_stage_family::feed_forward_down:
      return block != nullptr ? block->feed_forward_down.tensor : nullptr;
    case llama::detail::quantized_stage_family::feed_forward_up:
      return block != nullptr ? block->feed_forward_up.tensor : nullptr;
  }
  return nullptr;
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

std::string_view quantized_stage_family_name(const quantized_stage_family family) noexcept {
  switch (family) {
    case quantized_stage_family::token_embedding:
      return "token_embedding";
    case quantized_stage_family::output_norm:
      return "output_norm";
    case quantized_stage_family::output:
      return "output";
    case quantized_stage_family::attention_norm:
      return "attention_norm";
    case quantized_stage_family::attention_q:
      return "attention_q";
    case quantized_stage_family::attention_k:
      return "attention_k";
    case quantized_stage_family::attention_v:
      return "attention_v";
    case quantized_stage_family::attention_output:
      return "attention_output";
    case quantized_stage_family::feed_forward_norm:
      return "feed_forward_norm";
    case quantized_stage_family::feed_forward_gate:
      return "feed_forward_gate";
    case quantized_stage_family::feed_forward_down:
      return "feed_forward_down";
    case quantized_stage_family::feed_forward_up:
      return "feed_forward_up";
  }
  return "unknown_stage";
}

std::string_view quantized_contract_kind_name(const quantized_contract_kind kind) noexcept {
  switch (kind) {
    case quantized_contract_kind::native_quantized:
      return "native_quantized";
    case quantized_contract_kind::approved_dense_f32_by_contract:
      return "approved_dense_f32_by_contract";
    case quantized_contract_kind::disallowed_fallback:
      return "disallowed_fallback";
    case quantized_contract_kind::explicit_no_claim:
      return "explicit_no_claim";
  }
  return "unknown_contract";
}

std::string_view tensor_type_name(const int32_t tensor_type) noexcept {
  switch (static_cast<emel::kernel::event::dtype>(tensor_type)) {
    case emel::kernel::event::dtype::f32:
      return "f32";
    case emel::kernel::event::dtype::q2_k:
      return "q2_k";
    case emel::kernel::event::dtype::q3_k:
      return "q3_k";
    case emel::kernel::event::dtype::q6_k:
      return "q6_k";
    default:
      break;
  }

  return static_cast<uint8_t>(tensor_type) == emel::kernel::detail::dtype_q4_0 ? "q4_0"
                                                                                : "unknown";
}

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

quantized_path_audit build_quantized_path_audit(const execution_view & execution) noexcept {
  quantized_path_audit audit = {};

  for (size_t idx = 0; idx < k_stage_families.size(); ++idx) {
    const auto family = k_stage_families[idx];
    quantized_stage_audit stage{};
    stage.family = family;

    if (family == quantized_stage_family::token_embedding ||
        family == quantized_stage_family::output_norm ||
        family == quantized_stage_family::output) {
      const auto * tensor = stage_tensor(execution, nullptr, family);
      stage.tensor_type = tensor != nullptr ? tensor->type : -1;
      stage.contract = classify_contract(family, stage.tensor_type, true);
      audit.stages[idx] = stage;
      continue;
    }

    bool first = true;
    int32_t first_type = -1;
    bool consistent = execution.block_count > 0;
    for (int32_t block_index = 0; block_index < execution.block_count; ++block_index) {
      block_view block{};
      if (lookup_block_view(execution, block_index, block) !=
          emel::error::cast(emel::model::loader::error::none)) {
        consistent = false;
        first_type = -1;
        break;
      }

      const auto * tensor = stage_tensor(execution, &block, family);
      const int32_t tensor_type = tensor != nullptr ? tensor->type : -1;
      if (first) {
        first_type = tensor_type;
        first = false;
      } else if (tensor_type != first_type) {
        consistent = false;
      }
    }

    stage.tensor_type = first_type;
    stage.consistent_across_layers = consistent;
    stage.contract = classify_contract(family, stage.tensor_type, consistent);
    audit.stages[idx] = stage;
  }

  return audit;
}

}  // namespace llama::detail

}  // namespace emel::model
