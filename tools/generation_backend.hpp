#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/generator/events.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/llama/detail.hpp"

namespace emel::tools::generation_backend {

struct dequantized_matrix {
  std::vector<float> values = {};
  int32_t rows = 0;
  int32_t cols = 0;
};

struct block_weights {
  std::vector<float> attention_norm = {};
  dequantized_matrix attention_q = {};
  dequantized_matrix attention_k = {};
  dequantized_matrix attention_v = {};
  dequantized_matrix attention_output = {};
  std::vector<float> feed_forward_norm = {};
  dequantized_matrix feed_forward_gate = {};
  dequantized_matrix feed_forward_down = {};
  dequantized_matrix feed_forward_up = {};
};

struct native_backend {
  const emel::model::data * model = nullptr;
  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan prefill_plan = {};
  emel::model::llama::detail::step_plan decode_plan = {};

  dequantized_matrix token_embedding = {};
  std::vector<float> output_norm = {};
  dequantized_matrix output = {};
  std::vector<block_weights> blocks = {};

  int32_t n_vocab = 0;
  int32_t n_embd = 0;
  int32_t n_head = 0;
  int32_t n_head_kv = 0;
  int32_t n_layer = 0;
  int32_t n_ctx = 0;
  int32_t n_rot = 0;
  int32_t head_dim = 0;
  int32_t head_dim_kv = 0;
  int32_t n_rep = 0;
  float rms_epsilon = 1.0e-5f;
  float rope_freq_base = 10000.0f;

  std::vector<float> key_cache = {};
  std::vector<float> value_cache = {};
  int32_t kv_cache_tokens = 0;

  std::vector<int32_t> bound_tokens = {};
  std::vector<int32_t> bound_positions = {};
  std::vector<float> bound_logits = {};

  std::vector<float> hidden = {};
  std::vector<float> norm = {};
  std::vector<float> q = {};
  std::vector<float> k = {};
  std::vector<float> v = {};
  std::vector<float> attn_scores = {};
  std::vector<float> attn_probs = {};
  std::vector<float> attn_ctx = {};
  std::vector<float> projected = {};
  std::vector<float> gate = {};
  std::vector<float> up = {};
  std::vector<float> ffn_hidden = {};
  bool bound_ready = false;
};

emel::error::type prepare(native_backend & backend, const emel::model::data & model_data) noexcept;

bool validate(const emel::graph::processor::event::execute & request, int32_t * err_out);
bool prepare_graph(const emel::graph::processor::event::execute & request,
                   bool * reused_out,
                   int32_t * err_out);
bool alloc_graph(const emel::graph::processor::event::execute & request, int32_t * err_out);
bool bind_inputs(const emel::graph::processor::event::execute & request, int32_t * err_out);
bool run_kernel(const emel::graph::processor::event::execute & request, int32_t * err_out);
bool extract_outputs(const emel::graph::processor::event::execute & request,
                     int32_t * outputs_out,
                     int32_t * err_out);

}  // namespace emel::tools::generation_backend
