#include "emel/model/data.hpp"

#include <array>
#include <cstdio>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/model/architecture/detail.hpp"
#include "emel/model/gemma4/detail.hpp"
#include "emel/model/lfm2/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/omniembed/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model {

namespace {

constexpr std::string_view k_qwen3_architecture = "qwen3";
constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "output_norm.weight";
constexpr std::string_view k_lfm2_output_norm_name = "token_embd_norm.weight";
constexpr std::string_view k_output_name = "output.weight";
constexpr uint32_t k_lfm2_attention_block_tensor_count = 11u;
constexpr uint32_t k_lfm2_conv_block_tensor_count = 8u;
constexpr std::array<int32_t, 6> k_lfm2_attention_layers = {2, 5, 8, 10, 12, 14};

bool uses_qwen3_block_contract(const std::string_view architecture) noexcept {
  return architecture == k_qwen3_architecture;
}

bool uses_lfm2_block_contract(const std::string_view architecture) noexcept {
  return emel::model::lfm2::detail::is_execution_architecture(architecture);
}

bool uses_gemma4_block_contract(const std::string_view architecture) noexcept {
  return emel::model::gemma4::detail::is_execution_architecture(architecture);
}

bool uses_tied_output_contract(const std::string_view architecture) noexcept {
  return uses_qwen3_block_contract(architecture) ||
         uses_lfm2_block_contract(architecture) ||
         uses_gemma4_block_contract(architecture);
}

bool uses_attention_qk_norm(const std::string_view architecture) noexcept {
  return uses_qwen3_block_contract(architecture) ||
         uses_lfm2_block_contract(architecture) ||
         uses_gemma4_block_contract(architecture);
}

bool is_lfm2_attention_layer(const int32_t block_index) noexcept {
  for (const int32_t candidate : k_lfm2_attention_layers) {
    if (candidate == block_index) {
      return true;
    }
  }
  return false;
}

bool is_gemma4_shared_kv_layer(const data & model_data, const int32_t block_index) noexcept {
  const int32_t shared_kv_layers = model_data.params.attention_shared_kv_layers;
  if (shared_kv_layers <= 0 || block_index < 0 || block_index >= model_data.n_layers) {
    return false;
  }

  const int32_t first_shared = model_data.n_layers - shared_kv_layers;
  return first_shared >= 0 && block_index >= first_shared;
}

uint32_t block_tensor_count_for_architecture(const std::string_view architecture) noexcept {
  return uses_qwen3_block_contract(architecture)
             ? llama::detail::k_qwen3_block_tensor_count
             : llama::detail::k_block_tensor_count;
}

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

bool assign_output_view(const data & model_data,
                        const std::string_view architecture,
                        const llama::detail::tensor_view & token_embedding,
                        llama::detail::tensor_view & output_out) noexcept {
  if (assign_tensor_view(model_data, k_output_name, output_out)) {
    return true;
  }

  if (!uses_tied_output_contract(architecture) || token_embedding.tensor == nullptr) {
    return false;
  }

  output_out = token_embedding;
  return true;
}

bool assign_output_norm_view(const data & model_data,
                             const std::string_view architecture,
                             llama::detail::tensor_view & output_norm_out) noexcept {
  if (assign_tensor_view(model_data, k_output_norm_name, output_norm_out)) {
    return true;
  }

  return uses_lfm2_block_contract(architecture) &&
         assign_tensor_view(model_data, k_lfm2_output_norm_name, output_norm_out);
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
        llama::detail::quantized_stage_family::attention_q_norm,
        llama::detail::quantized_stage_family::attention_k_norm,
        llama::detail::quantized_stage_family::attention_output,
        llama::detail::quantized_stage_family::feed_forward_norm,
        llama::detail::quantized_stage_family::feed_forward_gate,
        llama::detail::quantized_stage_family::feed_forward_down,
        llama::detail::quantized_stage_family::feed_forward_up,
    };

bool is_supported_quantized_type(const int32_t tensor_type) noexcept {
  return ::emel::kernel::detail::is_native_quantized_dtype(static_cast<uint8_t>(tensor_type));
}

bool is_f32_type(const int32_t tensor_type) noexcept {
  return static_cast<uint8_t>(tensor_type) == ::emel::kernel::detail::dtype_f32;
}

bool is_vector_dequant_stage(const llama::detail::quantized_stage_family family) noexcept {
  return family == llama::detail::quantized_stage_family::token_embedding ||
         family == llama::detail::quantized_stage_family::output_norm ||
         family == llama::detail::quantized_stage_family::attention_norm ||
         family == llama::detail::quantized_stage_family::attention_q_norm ||
         family == llama::detail::quantized_stage_family::attention_k_norm ||
         family == llama::detail::quantized_stage_family::feed_forward_norm;
}

bool stage_applies_to_block(const llama::detail::block_view & block,
                            const llama::detail::quantized_stage_family family) noexcept {
  switch (family) {
    case llama::detail::quantized_stage_family::attention_norm:
      return block.attention_norm.tensor != nullptr;
    case llama::detail::quantized_stage_family::attention_q:
      return block.uses_attention && block.attention_q.tensor != nullptr;
    case llama::detail::quantized_stage_family::attention_k:
      return block.uses_attention && block.attention_k.tensor != nullptr;
    case llama::detail::quantized_stage_family::attention_v:
      return block.uses_attention && block.attention_v.tensor != nullptr;
    case llama::detail::quantized_stage_family::attention_q_norm:
      return block.uses_attention && block.attention_q_norm.tensor != nullptr;
    case llama::detail::quantized_stage_family::attention_k_norm:
      return block.uses_attention && block.attention_k_norm.tensor != nullptr;
    case llama::detail::quantized_stage_family::attention_output:
      return block.uses_attention && block.attention_output.tensor != nullptr;
    case llama::detail::quantized_stage_family::feed_forward_norm:
      return block.feed_forward_norm.tensor != nullptr;
    case llama::detail::quantized_stage_family::feed_forward_gate:
      return block.feed_forward_gate.tensor != nullptr;
    case llama::detail::quantized_stage_family::feed_forward_down:
      return block.feed_forward_down.tensor != nullptr;
    case llama::detail::quantized_stage_family::feed_forward_up:
      return block.feed_forward_up.tensor != nullptr;
    default:
      return false;
  }
}

llama::detail::quantized_contract_kind classify_stage_contract(
    const llama::detail::quantized_stage_family family,
    const bool saw_applicable_tensor,
    const bool all_supported_quantized,
    const bool all_vector_approved,
    const bool any_f32) noexcept {
  if (!saw_applicable_tensor) {
    return llama::detail::quantized_contract_kind::not_applicable;
  }

  if (is_vector_dequant_stage(family)) {
    return all_vector_approved
               ? llama::detail::quantized_contract_kind::approved_dense_f32_by_contract
               : llama::detail::quantized_contract_kind::explicit_no_claim;
  }

  if (all_supported_quantized) {
    return llama::detail::quantized_contract_kind::native_quantized;
  }

  return any_f32 ? llama::detail::quantized_contract_kind::disallowed_fallback
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
    case llama::detail::quantized_stage_family::attention_q_norm:
      return block != nullptr ? block->attention_q_norm.tensor : nullptr;
    case llama::detail::quantized_stage_family::attention_k_norm:
      return block != nullptr ? block->attention_k_norm.tensor : nullptr;
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

std::string_view metadata_string_view(
    const data::metadata & metadata,
    const data::metadata::string_view & value) noexcept {
  const size_t begin = static_cast<size_t>(value.offset);
  const size_t length = static_cast<size_t>(value.length);
  if (length == 0u || begin + length > metadata.blob_bytes_used || begin + length > metadata.blob.size()) {
    return {};
  }

  return std::string_view{metadata.blob.data() + begin, length};
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

bool is_supported_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::resolve_architecture(
             architecture, emel::model::default_architecture_span()) != nullptr;
}

bool is_lfm2_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::lfm2::detail::is_execution_architecture(architecture);
}

bool is_gemma4_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::gemma4::detail::is_execution_architecture(architecture);
}

bool is_omniembed_execution_architecture(const std::string_view architecture) noexcept {
  return emel::model::omniembed::detail::is_execution_architecture(architecture);
}

emel::error::type validate_execution_contract(const data & model_data) noexcept {
  const auto architecture = emel::model::architecture_name_view(model_data);
  const auto * resolved = emel::model::resolve_architecture(
      architecture, emel::model::default_architecture_span());
  if (resolved == nullptr || resolved->validate_data == nullptr) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return resolved->validate_data(model_data);
}

namespace llama::detail {

bool has_tensor_named(const emel::model::data & model_data, const std::string_view name) noexcept {
  const auto * tensor = find_tensor_by_name(model_data, name);
  return tensor != nullptr && tensor_has_storage(*tensor);
}

bool require_block_tensor(const emel::model::data & model_data,
                          const int32_t block_index,
                          const std::string_view suffix) noexcept {
  std::array<char, 64> buffer = {};
  std::string_view name = {};
  return make_block_tensor_name(block_index, suffix, buffer, name) &&
         has_tensor_named(model_data, name);
}

bool reject_block_tensor(const emel::model::data & model_data,
                         const int32_t block_index,
                         const std::string_view suffix) noexcept {
  std::array<char, 64> buffer = {};
  std::string_view name = {};
  return !make_block_tensor_name(block_index, suffix, buffer, name) ||
         !has_tensor_named(model_data, name);
}

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
    case quantized_stage_family::attention_q_norm:
      return "attention_q_norm";
    case quantized_stage_family::attention_k_norm:
      return "attention_k_norm";
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
    case quantized_contract_kind::not_applicable:
      return "not_applicable";
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
    case emel::kernel::event::dtype::q4_k:
      return "q4_k";
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
  const auto architecture = architecture_name_view(*execution.model);
  if (!is_supported_execution_architecture(architecture)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  std::array<char, 64> buffer = {};
  std::string_view name = {};
  const auto bind = [&](const std::string_view suffix, tensor_view & tensor_out) {
    return make_block_tensor_name(block_index, suffix, buffer, name) &&
           assign_tensor_view(*execution.model, name, tensor_out);
  };

  const bool common_ok =
      bind("attn_norm.weight", block_out.attention_norm) &&
      bind("ffn_norm.weight", block_out.feed_forward_norm) &&
      bind("ffn_gate.weight", block_out.feed_forward_gate) &&
      bind("ffn_down.weight", block_out.feed_forward_down) &&
      bind("ffn_up.weight", block_out.feed_forward_up);
  if (!common_ok) {
    block_out = {};
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  block_out.uses_attention =
      !uses_lfm2_block_contract(architecture) || is_lfm2_attention_layer(block_index);
  if (!block_out.uses_attention) {
    const bool shortconv_ok =
        bind("shortconv.conv.weight", block_out.shortconv_conv) &&
        bind("shortconv.in_proj.weight", block_out.shortconv_in_proj) &&
        bind("shortconv.out_proj.weight", block_out.shortconv_out_proj);
    if (!shortconv_ok) {
      block_out = {};
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    return emel::error::cast(emel::model::loader::error::none);
  }

  const bool requires_qk_norm = uses_attention_qk_norm(architecture);
  const bool uses_shared_gemma4_kv =
      uses_gemma4_block_contract(architecture) &&
      is_gemma4_shared_kv_layer(*execution.model, block_index);
  const bool attention_common_ok =
      bind("attn_q.weight", block_out.attention_q) &&
      bind("attn_k.weight", block_out.attention_k) &&
      (!requires_qk_norm || bind("attn_q_norm.weight", block_out.attention_q_norm)) &&
      (!requires_qk_norm || bind("attn_k_norm.weight", block_out.attention_k_norm)) &&
      bind("attn_output.weight", block_out.attention_output);
  if (!attention_common_ok) {
    block_out = {};
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const bool value_ok =
      bind("attn_v.weight", block_out.attention_v) ||
      (uses_shared_gemma4_kv && block_out.attention_k.tensor != nullptr &&
       ((block_out.attention_v = block_out.attention_k), true));
  if (!value_ok) {
    block_out = {};
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type build_execution_view(const emel::model::data & model_data,
                                       execution_view & view_out) noexcept {
  view_out = {};
  const auto architecture = architecture_name_view(model_data);

  if (!is_supported_execution_architecture(architecture) ||
      model_data.n_tensors == 0u ||
      model_data.n_layers <= 0 ||
      model_data.params.n_embd <= 0 ||
      model_data.params.n_ctx <= 0) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  execution_view view = {};
  view.model = &model_data;
  view.block_count = model_data.n_layers;

  if (!assign_tensor_view(model_data, k_token_embedding_name, view.token_embedding) ||
      !assign_output_norm_view(model_data, architecture, view.output_norm) ||
      !assign_output_view(model_data, architecture, view.token_embedding, view.output)) {
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
  const auto architecture = architecture_name_view(*execution.model);
  if (uses_lfm2_block_contract(architecture)) {
    uint32_t tensor_count = k_global_tensor_count;
    for (int32_t block_index = 0; block_index < execution.block_count; ++block_index) {
      tensor_count += is_lfm2_attention_layer(block_index) ? k_lfm2_attention_block_tensor_count
                                                           : k_lfm2_conv_block_tensor_count;
    }
    topology_out.tensor_count = tensor_count;
  } else if (uses_gemma4_block_contract(architecture)) {
    uint32_t tensor_count = k_global_tensor_count;
    for (int32_t block_index = 0; block_index < execution.block_count; ++block_index) {
      tensor_count += is_gemma4_shared_kv_layer(*execution.model, block_index)
                          ? k_gemma4_shared_kv_block_tensor_count
                          : k_gemma4_dedicated_kv_block_tensor_count;
    }
    topology_out.tensor_count = tensor_count;
  } else {
    topology_out.tensor_count =
        k_global_tensor_count +
        (static_cast<uint32_t>(execution.block_count) *
         block_tensor_count_for_architecture(architecture));
  }

  topology_out.node_count = topology_out.tensor_count;
  topology_out.bytes_per_tensor = sizeof(float);
  topology_out.workspace_capacity_bytes =
      static_cast<uint64_t>(topology_out.tensor_count) *
      static_cast<uint64_t>(execution.model->params.n_embd) *
      sizeof(float);
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
      stage.contract = classify_stage_contract(
          family,
          tensor != nullptr,
          is_supported_quantized_type(stage.tensor_type),
          is_supported_quantized_type(stage.tensor_type) || is_f32_type(stage.tensor_type),
          is_f32_type(stage.tensor_type));
      audit.stages[idx] = stage;
      continue;
    }

    bool first = true;
    int32_t first_type = -1;
    bool consistent = execution.block_count > 0;
    bool saw_applicable_tensor = false;
    bool all_supported_quantized = true;
    bool all_vector_approved = true;
    bool any_f32 = false;
    for (int32_t block_index = 0; block_index < execution.block_count; ++block_index) {
      block_view block{};
      if (lookup_block_view(execution, block_index, block) !=
          emel::error::cast(emel::model::loader::error::none)) {
        consistent = false;
        first_type = -1;
        break;
      }

      if (!stage_applies_to_block(block, family)) {
        continue;
      }

      const auto * tensor = stage_tensor(execution, &block, family);
      const int32_t tensor_type = tensor != nullptr ? tensor->type : -1;
      const bool supported_quantized = is_supported_quantized_type(tensor_type);
      const bool f32_type = is_f32_type(tensor_type);
      if (first) {
        first_type = tensor_type;
        first = false;
      } else if (tensor_type != first_type) {
        consistent = false;
      }
      saw_applicable_tensor = true;
      all_supported_quantized = all_supported_quantized && supported_quantized;
      all_vector_approved = all_vector_approved && (supported_quantized || f32_type);
      any_f32 = any_f32 || f32_type;
    }

    stage.tensor_type = first_type;
    stage.consistent_across_layers = consistent && saw_applicable_tensor;
    stage.contract = classify_stage_contract(
        family, saw_applicable_tensor, all_supported_quantized, all_vector_approved, any_f32);
    audit.stages[idx] = stage;
  }

  return audit;
}

}  // namespace llama::detail

}  // namespace emel::model
