#include "generation_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/model/loader/errors.hpp"
#include "ggml.h"

namespace emel::tools::generation_backend {

namespace {

using tensor_record = emel::model::data::tensor_record;
using step_plan = emel::model::llama::detail::step_plan;
using step_kind = emel::model::llama::detail::step_kind;

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_invalid = 1;

template <class tensor_type>
void fill_default_nb(tensor_type & tensor) {
  constexpr uint64_t elem_size = sizeof(float);
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

emel::kernel::event::tensor_view make_src_view(const float * data,
                                               const uint64_t ne0,
                                               const uint64_t ne1 = 1u) {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(tensor);
  return tensor;
}

emel::kernel::event::tensor_view_mut make_dst_view(float * data,
                                                   const uint64_t ne0,
                                                   const uint64_t ne1 = 1u) {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(tensor);
  return tensor;
}

bool dequantize_tensor_rows(const tensor_record & tensor, dequantized_matrix & out) {
  out = {};
  if (tensor.data == nullptr || tensor.n_dims <= 0 || tensor.dims[0] <= 0) {
    return false;
  }

  const auto * traits = ggml_get_type_traits(static_cast<ggml_type>(tensor.type));
  if (traits == nullptr) {
    return false;
  }

  const int32_t cols = static_cast<int32_t>(tensor.dims[0]);
  const int32_t rows = tensor.n_dims > 1 ? static_cast<int32_t>(tensor.dims[1]) : 1;
  if (cols <= 0 || rows <= 0) {
    return false;
  }

  const size_t row_bytes = ggml_row_size(static_cast<ggml_type>(tensor.type), cols);
  if (row_bytes == 0u) {
    return false;
  }

  out.rows = rows;
  out.cols = cols;
  out.values.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));

  const auto * src = static_cast<const uint8_t *>(tensor.data);
  for (int32_t row = 0; row < rows; ++row) {
    float * dst_row =
        out.values.data() + (static_cast<size_t>(row) * static_cast<size_t>(cols));
    const auto * src_row = src + (static_cast<size_t>(row) * row_bytes);
    if (traits->to_float != nullptr) {
      traits->to_float(src_row, dst_row, cols);
    } else if (!traits->is_quantized && traits->type_size == sizeof(float)) {
      std::memcpy(dst_row, src_row, static_cast<size_t>(cols) * sizeof(float));
    } else {
      return false;
    }
  }
  return true;
}

bool dequantize_tensor_vector(const tensor_record & tensor, std::vector<float> & out) {
  dequantized_matrix matrix = {};
  if (!dequantize_tensor_rows(tensor, matrix) || matrix.rows != 1 || matrix.cols <= 0) {
    return false;
  }
  out = std::move(matrix.values);
  return true;
}

bool matmul_vector(const dequantized_matrix & matrix,
                   std::span<const float> input,
                   std::span<float> output) {
  if (matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix.values.data(),
                            static_cast<uint64_t>(matrix.cols),
                            static_cast<uint64_t>(matrix.rows)),
      .src1 = make_src_view(input.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(input.size())),
      .dst = make_dst_view(output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
      .nth = 1,
  };
  return emel::kernel::detail::run_mul_mat(ev);
}

bool rms_norm(std::span<const float> input,
              std::span<const float> weight,
              const float epsilon,
              std::span<float> output) {
  if (input.size() != weight.size() || input.size() != output.size() || input.empty()) {
    return false;
  }

  double square_sum = 0.0;
  for (const float value : input) {
    square_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const float scale =
      1.0f / std::sqrt(static_cast<float>(square_sum / static_cast<double>(input.size())) + epsilon);
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i] * scale * weight[i];
  }
  return true;
}

void apply_rope(std::span<float> vector,
                const int32_t head_count,
                const int32_t head_dim,
                const int32_t n_rot,
                const int32_t position,
                const float rope_freq_base) {
  const int32_t rot_dim = std::min(n_rot, head_dim);
  if (head_count <= 0 || head_dim <= 1 || rot_dim <= 1) {
    return;
  }

  for (int32_t head = 0; head < head_count; ++head) {
    float * head_ptr = vector.data() + (static_cast<size_t>(head) * static_cast<size_t>(head_dim));
    for (int32_t dim = 0; dim + 1 < rot_dim; dim += 2) {
      const float inv_freq =
          std::pow(rope_freq_base, -static_cast<float>(dim) / static_cast<float>(rot_dim));
      const float theta = static_cast<float>(position) * inv_freq;
      const float cos_theta = std::cos(theta);
      const float sin_theta = std::sin(theta);
      const float x0 = head_ptr[dim];
      const float x1 = head_ptr[dim + 1];
      head_ptr[dim] = x0 * cos_theta - x1 * sin_theta;
      head_ptr[dim + 1] = x0 * sin_theta + x1 * cos_theta;
    }
  }
}

float silu(const float value) {
  return value / (1.0f + std::exp(-value));
}

size_t layer_cache_offset(const native_backend & backend,
                          const int32_t layer,
                          const int32_t position,
                          const int32_t kv_dim) {
  return ((static_cast<size_t>(layer) * static_cast<size_t>(backend.n_ctx)) +
          static_cast<size_t>(position)) *
         static_cast<size_t>(kv_dim);
}

bool check_backend(const native_backend * backend, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = k_error_ok;
  }
  if (backend == nullptr ||
      backend->model == nullptr ||
      backend->n_embd <= 0 ||
      backend->n_head <= 0 ||
      backend->n_head_kv <= 0 ||
      backend->n_layer <= 0 ||
      backend->n_vocab <= 0 ||
      backend->n_ctx <= 0 ||
      backend->head_dim <= 0 ||
      backend->head_dim_kv <= 0 ||
      backend->blocks.size() != static_cast<size_t>(backend->n_layer)) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }
  return true;
}

const step_plan * request_plan(const emel::graph::processor::event::execute & request,
                               int32_t * err_out) {
  const auto * plan = static_cast<const step_plan *>(request.step_plan);
  if (plan == nullptr || plan->graph == nullptr || plan->graph->execution == nullptr) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return nullptr;
  }
  return plan;
}

bool store_bound_request(native_backend & backend,
                         const emel::graph::processor::event::execute & request,
                         int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  if (io == nullptr ||
      io->token_ids == nullptr ||
      io->token_count <= 0 ||
      request.positions == nullptr ||
      request.positions_count != io->token_count) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  backend.bound_tokens.assign(io->token_ids, io->token_ids + io->token_count);
  backend.bound_positions.assign(request.positions, request.positions + request.positions_count);
  backend.bound_ready = true;
  return true;
}

bool compute_attention(native_backend & backend,
                       const int32_t layer_index,
                       const int32_t position_limit,
                       const std::span<const float> q_vector) {
  const int32_t head_count = backend.n_head;
  const int32_t kv_head_count = backend.n_head_kv;
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const int32_t kv_dim = kv_head_count * kv_head_dim;
  const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::fill(backend.attn_ctx.begin(), backend.attn_ctx.end(), 0.0f);

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    float max_score = -std::numeric_limits<float>::infinity();
    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset = layer_cache_offset(backend, layer_index, position, kv_dim) + kv_offset;
      float score = 0.0f;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        score += q_vector[q_offset + static_cast<size_t>(dim)] *
                 backend.key_cache[cache_offset + static_cast<size_t>(dim)];
      }
      score *= inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
      max_score = std::max(max_score, score);
    }

    float score_sum = 0.0f;
    for (int32_t position = 0; position < position_limit; ++position) {
      const float prob = std::exp(backend.attn_scores[static_cast<size_t>(position)] - max_score);
      backend.attn_probs[static_cast<size_t>(position)] = prob;
      score_sum += prob;
    }

    for (int32_t position = 0; position < position_limit; ++position) {
      const float weight = backend.attn_probs[static_cast<size_t>(position)] / score_sum;
      const size_t cache_offset = layer_cache_offset(backend, layer_index, position, kv_dim) + kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * backend.value_cache[cache_offset + static_cast<size_t>(dim)];
      }
    }
  }

  return true;
}

bool run_layer(native_backend & backend,
               const int32_t layer_index,
               const int32_t position) {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!rms_norm(backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector(block.attention_q, backend.norm, backend.q) ||
      !matmul_vector(block.attention_k, backend.norm, backend.k) ||
      !matmul_vector(block.attention_v, backend.norm, backend.v)) {
    return false;
  }

  apply_rope(backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
  apply_rope(backend.k,
             backend.n_head_kv,
             backend.head_dim_kv,
             backend.n_rot,
             position,
             backend.rope_freq_base);

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset = layer_cache_offset(backend, layer_index, position, kv_dim);
  std::copy(backend.k.begin(),
            backend.k.begin() + kv_dim,
            backend.key_cache.begin() + static_cast<std::ptrdiff_t>(cache_offset));
  std::copy(backend.v.begin(),
            backend.v.begin() + kv_dim,
            backend.value_cache.begin() + static_cast<std::ptrdiff_t>(cache_offset));

  if (!compute_attention(backend, layer_index, position + 1, backend.q) ||
      !matmul_vector(block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!rms_norm(backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector(block.feed_forward_gate, backend.norm, backend.gate) ||
      !matmul_vector(block.feed_forward_up, backend.norm, backend.up)) {
    return false;
  }

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = silu(backend.gate[idx]) * backend.up[idx];
  }

  if (!matmul_vector(block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

bool compute_logits(native_backend & backend) {
  if (!rms_norm(backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector(backend.output, backend.norm, backend.bound_logits)) {
    return false;
  }
  return true;
}

bool run_prefill(native_backend & backend) {
  backend.kv_cache_tokens = 0;

  for (size_t token_index = 0; token_index < backend.bound_tokens.size(); ++token_index) {
    const int32_t token_id = backend.bound_tokens[token_index];
    const int32_t position = backend.bound_positions[token_index];
    if (token_id < 0 ||
        token_id >= backend.token_embedding.rows ||
        position < 0 ||
        position >= backend.n_ctx) {
      return false;
    }

    const size_t embedding_offset =
        static_cast<size_t>(token_id) * static_cast<size_t>(backend.token_embedding.cols);
    std::copy_n(backend.token_embedding.values.data() + static_cast<std::ptrdiff_t>(embedding_offset),
                static_cast<size_t>(backend.n_embd),
                backend.hidden.data());

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return compute_logits(backend);
}

bool run_decode(native_backend & backend, const emel::graph::processor::event::execute & request) {
  if (backend.bound_tokens.size() != 1u ||
      backend.bound_positions.size() != 1u ||
      request.kv_tokens < 0 ||
      backend.kv_cache_tokens != request.kv_tokens) {
    return false;
  }

  const int32_t token_id = backend.bound_tokens[0];
  const int32_t position = backend.bound_positions[0];
  if (token_id < 0 ||
      token_id >= backend.token_embedding.rows ||
      position < 0 ||
      position >= backend.n_ctx) {
    return false;
  }

  const size_t embedding_offset =
      static_cast<size_t>(token_id) * static_cast<size_t>(backend.token_embedding.cols);
  std::copy_n(backend.token_embedding.values.data() + static_cast<std::ptrdiff_t>(embedding_offset),
              static_cast<size_t>(backend.n_embd),
              backend.hidden.data());

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!run_layer(backend, layer, position)) {
      return false;
    }
  }
  backend.kv_cache_tokens = position + 1;
  return compute_logits(backend);
}

}  // namespace

emel::error::type prepare(native_backend & backend, const emel::model::data & model_data) noexcept {
  backend = {};

  if (emel::model::llama::detail::build_execution_view(model_data, backend.execution) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::model::llama::detail::build_topology(backend.execution, backend.topology) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::model::llama::detail::build_step_plans(
          backend.topology,
          backend.prefill_plan,
          backend.decode_plan) != emel::error::cast(emel::model::loader::error::none)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  backend.model = &model_data;
  backend.n_vocab = model_data.params.n_vocab;
  backend.n_embd = model_data.params.n_embd;
  backend.n_head = model_data.params.n_head;
  backend.n_head_kv = model_data.params.n_head_kv > 0 ? model_data.params.n_head_kv : model_data.params.n_head;
  backend.n_layer = backend.execution.block_count;
  backend.n_ctx = model_data.params.n_ctx;
  backend.n_rot = model_data.params.n_rot > 0 ? model_data.params.n_rot : 0;
  backend.rms_epsilon = model_data.params.attention_layer_norm_rms_epsilon > 0.0f
                            ? model_data.params.attention_layer_norm_rms_epsilon
                            : 1.0e-5f;
  backend.rope_freq_base =
      model_data.params.rope_freq_base > 0.0f ? model_data.params.rope_freq_base : 10000.0f;

  if (backend.n_vocab <= 0 ||
      backend.n_embd <= 0 ||
      backend.n_head <= 0 ||
      backend.n_head_kv <= 0 ||
      backend.n_layer <= 0 ||
      backend.n_ctx <= 0 ||
      backend.n_embd % backend.n_head != 0) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  backend.head_dim = backend.n_embd / backend.n_head;

  const bool token_embedding_ok =
      dequantize_tensor_rows(*backend.execution.token_embedding.tensor, backend.token_embedding);
  const bool output_norm_ok =
      dequantize_tensor_vector(*backend.execution.output_norm.tensor, backend.output_norm);
  const bool output_ok =
      dequantize_tensor_rows(*backend.execution.output.tensor, backend.output);
  if (!token_embedding_ok || !output_norm_ok || !output_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  if (backend.token_embedding.cols != backend.n_embd ||
      backend.token_embedding.rows < backend.n_vocab ||
      static_cast<int32_t>(backend.output_norm.size()) != backend.n_embd ||
      backend.output.cols != backend.n_embd ||
      backend.output.rows < backend.n_vocab) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  backend.blocks.resize(static_cast<size_t>(backend.n_layer));
  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    emel::model::llama::detail::block_view block = {};
    if (emel::model::llama::detail::lookup_block_view(backend.execution, layer, block) !=
        emel::error::cast(emel::model::loader::error::none)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    auto & weights = backend.blocks[static_cast<size_t>(layer)];
    if (!dequantize_tensor_vector(*block.attention_norm.tensor, weights.attention_norm) ||
        !dequantize_tensor_rows(*block.attention_q.tensor, weights.attention_q) ||
        !dequantize_tensor_rows(*block.attention_k.tensor, weights.attention_k) ||
        !dequantize_tensor_rows(*block.attention_v.tensor, weights.attention_v) ||
        !dequantize_tensor_rows(*block.attention_output.tensor, weights.attention_output) ||
        !dequantize_tensor_vector(*block.feed_forward_norm.tensor, weights.feed_forward_norm) ||
        !dequantize_tensor_rows(*block.feed_forward_gate.tensor, weights.feed_forward_gate) ||
        !dequantize_tensor_rows(*block.feed_forward_down.tensor, weights.feed_forward_down) ||
        !dequantize_tensor_rows(*block.feed_forward_up.tensor, weights.feed_forward_up)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  backend.head_dim_kv = backend.blocks[0].attention_k.rows / backend.n_head_kv;
  backend.n_rep = backend.n_head / backend.n_head_kv;

  if (backend.head_dim_kv <= 0 ||
      backend.n_rep <= 0 ||
      backend.blocks[0].attention_q.rows != backend.n_embd ||
      backend.blocks[0].attention_output.rows != backend.n_embd ||
      static_cast<int32_t>(backend.blocks[0].attention_norm.size()) != backend.n_embd ||
      static_cast<int32_t>(backend.blocks[0].feed_forward_norm.size()) != backend.n_embd ||
      backend.blocks[0].feed_forward_gate.cols != backend.n_embd ||
      backend.blocks[0].feed_forward_up.cols != backend.n_embd ||
      backend.blocks[0].feed_forward_down.rows != backend.n_embd) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  backend.key_cache.resize(static_cast<size_t>(backend.n_layer) *
                           static_cast<size_t>(backend.n_ctx) *
                           static_cast<size_t>(kv_dim));
  backend.value_cache.resize(static_cast<size_t>(backend.n_layer) *
                             static_cast<size_t>(backend.n_ctx) *
                             static_cast<size_t>(kv_dim));
  backend.bound_logits.resize(static_cast<size_t>(backend.n_vocab));
  backend.hidden.resize(static_cast<size_t>(backend.n_embd));
  backend.norm.resize(static_cast<size_t>(backend.n_embd));
  backend.q.resize(static_cast<size_t>(backend.n_embd));
  backend.k.resize(static_cast<size_t>(kv_dim));
  backend.v.resize(static_cast<size_t>(kv_dim));
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(static_cast<size_t>(backend.n_embd));
  backend.projected.resize(static_cast<size_t>(backend.n_embd));
  backend.gate.resize(static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));
  backend.up.resize(static_cast<size_t>(backend.blocks[0].feed_forward_up.rows));
  backend.ffn_hidden.resize(static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));

  return emel::error::cast(emel::model::loader::error::none);
}

bool validate(const emel::graph::processor::event::execute & request, int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  const auto * plan = request_plan(request, err_out);
  if (plan == nullptr || !check_backend(backend, err_out)) {
    return false;
  }

  if (request.expected_outputs != plan->expected_outputs ||
      io == nullptr ||
      io->logits == nullptr ||
      io->logits_capacity < backend->n_vocab ||
      request.positions == nullptr ||
      request.positions_count != io->token_count ||
      io->token_count <= 0) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  return true;
}

bool prepare_graph(const emel::graph::processor::event::execute &,
                   bool * reused_out,
                   int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = k_error_ok;
  }
  return true;
}

bool alloc_graph(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = k_error_ok;
  }
  return true;
}

bool bind_inputs(const emel::graph::processor::event::execute & request, int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  if (!check_backend(backend, err_out)) {
    return false;
  }
  return store_bound_request(*backend, request, err_out);
}

bool run_kernel(const emel::graph::processor::event::execute & request, int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  const auto * plan = request_plan(request, err_out);
  if (plan == nullptr || !check_backend(backend, err_out) || !backend->bound_ready) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  const bool ok = plan->kind == step_kind::prefill ? run_prefill(*backend)
                                                   : run_decode(*backend, request);
  if (err_out != nullptr) {
    *err_out = ok ? k_error_ok : k_error_invalid;
  }
  return ok;
}

bool extract_outputs(const emel::graph::processor::event::execute & request,
                     int32_t * outputs_out,
                     int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  if (!check_backend(backend, err_out) ||
      io == nullptr ||
      io->logits == nullptr ||
      io->logits_capacity < backend->n_vocab ||
      backend->bound_logits.size() != static_cast<size_t>(backend->n_vocab)) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  std::copy(backend->bound_logits.begin(),
            backend->bound_logits.end(),
            io->logits);
  for (int32_t idx = backend->n_vocab; idx < io->logits_capacity; ++idx) {
    io->logits[idx] = -1.0f;
  }
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = k_error_ok;
  }
  return true;
}

}  // namespace emel::tools::generation_backend
