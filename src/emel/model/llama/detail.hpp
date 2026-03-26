#pragma once

#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"

namespace emel::model::llama::detail {

constexpr uint32_t k_global_tensor_count = 3u;
constexpr uint32_t k_block_tensor_count = 8u;
constexpr uint32_t k_quantized_stage_family_count = 12u;

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct block_view {
  int32_t index = -1;
  tensor_view attention_norm = {};
  tensor_view attention_q = {};
  tensor_view attention_k = {};
  tensor_view attention_v = {};
  tensor_view attention_output = {};
  tensor_view feed_forward_norm = {};
  tensor_view feed_forward_gate = {};
  tensor_view feed_forward_down = {};
  tensor_view feed_forward_up = {};
};

struct execution_view {
  const emel::model::data * model = nullptr;
  tensor_view token_embedding = {};
  tensor_view output_norm = {};
  tensor_view output = {};
  int32_t block_count = 0;
};

struct topology {
  const execution_view * execution = nullptr;
  uint32_t node_count = 0u;
  uint32_t tensor_count = 0u;
  uint64_t bytes_per_tensor = 0u;
  uint64_t workspace_capacity_bytes = 0u;
};

enum class step_kind : uint8_t {
  prefill = 0,
  decode = 1,
};

struct step_plan {
  const topology * graph = nullptr;
  step_kind kind = step_kind::prefill;
  uint32_t node_count = 0u;
  uint32_t tensor_count = 0u;
  int32_t expected_outputs = 0;
  int32_t max_step_tokens = 0;
};

enum class quantized_stage_family : uint8_t {
  token_embedding = 0,
  output_norm,
  output,
  attention_norm,
  attention_q,
  attention_k,
  attention_v,
  attention_output,
  feed_forward_norm,
  feed_forward_gate,
  feed_forward_down,
  feed_forward_up,
};

enum class quantized_contract_kind : uint8_t {
  native_quantized = 0,
  approved_dense_f32_by_contract,
  disallowed_fallback,
  explicit_no_claim,
};

struct quantized_stage_audit {
  quantized_stage_family family = quantized_stage_family::token_embedding;
  int32_t tensor_type = 0;
  quantized_contract_kind contract = quantized_contract_kind::approved_dense_f32_by_contract;
  bool consistent_across_layers = true;
};

struct quantized_path_audit {
  std::array<quantized_stage_audit, k_quantized_stage_family_count> stages = {};
};

emel::error::type build_execution_view(const emel::model::data & model_data,
                                       execution_view & view_out) noexcept;
emel::error::type lookup_block_view(const execution_view & execution,
                                    int32_t block_index,
                                    block_view & block_out) noexcept;
emel::error::type build_topology(const execution_view & execution,
                                 topology & topology_out) noexcept;
emel::error::type build_step_plans(const topology & topology_in,
                                   step_plan & prefill_out,
                                   step_plan & decode_out) noexcept;
quantized_path_audit build_quantized_path_audit(const execution_view & execution) noexcept;
std::string_view quantized_stage_family_name(quantized_stage_family family) noexcept;
std::string_view quantized_contract_kind_name(quantized_contract_kind kind) noexcept;
std::string_view tensor_type_name(int32_t tensor_type) noexcept;

}  // namespace emel::model::llama::detail
