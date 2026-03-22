#pragma once

#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"

namespace emel::model::llama::detail {

constexpr uint32_t k_global_tensor_count = 3u;
constexpr uint32_t k_block_tensor_count = 8u;

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

}  // namespace emel::model::llama::detail
