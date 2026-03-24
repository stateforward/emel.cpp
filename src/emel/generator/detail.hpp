#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "emel/generator/events.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::generator::detail {

struct tensor_matrix {
  const emel::model::data::tensor_record * tensor = nullptr;
  int32_t rows = 0;
  int32_t cols = 0;
};

struct block_weights {
  std::vector<float> attention_norm = {};
  tensor_matrix attention_q = {};
  tensor_matrix attention_k = {};
  tensor_matrix attention_v = {};
  tensor_matrix attention_output = {};
  std::vector<float> feed_forward_norm = {};
  tensor_matrix feed_forward_gate = {};
  tensor_matrix feed_forward_down = {};
  tensor_matrix feed_forward_up = {};
};

struct native_backend {
  const emel::model::data * model = nullptr;
  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan prefill_plan = {};
  emel::model::llama::detail::step_plan decode_plan = {};
  emel::kernel::sm kernel = {};
  emel::kernel::kernel_kind kernel_kind = emel::kernel::kernel_kind::x86_64;
  uint64_t kernel_dispatch_calls = 0;
  uint64_t flash_attention_dispatch_calls = 0;

  tensor_matrix token_embedding = {};
  std::vector<float> output_norm = {};
  tensor_matrix output = {};
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

  std::vector<emel::graph::processor::event::lifecycle_tensor_binding> lifecycle_tensors = {};
  std::vector<int32_t> prefill_required_ids = {};
  std::vector<int32_t> prefill_publish_ids = {};
  std::vector<int32_t> prefill_release_ids = {};
  std::vector<int32_t> decode_required_ids = {};
  std::vector<int32_t> decode_publish_ids = {};
  std::vector<int32_t> decode_release_ids = {};
  emel::graph::processor::event::lifecycle_phase prefill_lifecycle_phase = {};
  emel::graph::processor::event::lifecycle_phase decode_lifecycle_phase = {};
  emel::graph::processor::event::lifecycle_manifest reserve_lifecycle = {};
  emel::graph::processor::event::lifecycle_manifest prefill_lifecycle = {};
  emel::graph::processor::event::lifecycle_manifest decode_lifecycle = {};
  int32_t input_tokens_tensor_id = -1;
  int32_t positions_tensor_id = -1;
  int32_t logits_tensor_id = -1;
  int32_t key_cache_tensor_id = -1;
  int32_t value_cache_tensor_id = -1;

  std::vector<int32_t> bound_tokens = {};
  std::vector<int32_t> bound_positions = {};
  std::vector<float> bound_logits = {};
  int32_t bound_token_count = 0;
  int32_t bound_position_count = 0;

  std::vector<float> hidden = {};
  std::vector<float> norm = {};
  std::vector<float> q = {};
  std::vector<float> q_attn = {};
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

namespace quant = emel::kernel::detail::quant;

namespace {

using tensor_record = emel::model::data::tensor_record;
using step_kind = emel::model::llama::detail::step_kind;
using step_plan = emel::model::llama::detail::step_plan;

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_invalid = 1;

constexpr emel::kernel::kernel_kind detect_host_kernel_kind() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
  return emel::kernel::kernel_kind::aarch64;
#elif defined(__x86_64__) || defined(_M_X64)
  return emel::kernel::kernel_kind::x86_64;
#elif defined(__wasm__)
  return emel::kernel::kernel_kind::wasm;
#else
  return emel::kernel::kernel_kind::x86_64;
#endif
}

template <class tensor_type>
void fill_default_nb(tensor_type & tensor) noexcept {
  constexpr uint64_t elem_size = sizeof(float);
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

inline emel::kernel::event::tensor_view make_src_view(const float * data,
                                                      const uint64_t ne0,
                                                      const uint64_t ne1 = 1u) noexcept {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(tensor);
  return tensor;
}

inline emel::kernel::event::tensor_view make_src_view_3d(const float * data,
                                                         const uint64_t ne0,
                                                         const uint64_t ne1,
                                                         const uint64_t ne2) noexcept {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, ne2, 1u};
  fill_default_nb(tensor);
  return tensor;
}

inline emel::kernel::event::tensor_view make_src_view_strided_3d(const float * data,
                                                                 const uint64_t ne0,
                                                                 const uint64_t ne1,
                                                                 const uint64_t ne2,
                                                                 const uint64_t nb1,
                                                                 const uint64_t nb2) noexcept {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, ne2, 1u};
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = nb1;
  tensor.nb[2] = nb2;
  return tensor;
}

inline emel::kernel::event::tensor_view_mut make_dst_view(float * data,
                                                          const uint64_t ne0,
                                                          const uint64_t ne1 = 1u) noexcept {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(tensor);
  return tensor;
}

inline emel::kernel::event::tensor_view_mut make_dst_view_3d(float * data,
                                                             const uint64_t ne0,
                                                             const uint64_t ne1,
                                                             const uint64_t ne2) noexcept {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {ne0, ne1, ne2, 1u};
  fill_default_nb(tensor);
  return tensor;
}

inline size_t row_storage_bytes(const tensor_record & tensor, const int32_t cols) noexcept {
  const uint8_t dtype = static_cast<uint8_t>(tensor.type);
  if (dtype == emel::kernel::detail::dtype_f32) {
    return static_cast<size_t>(cols) * sizeof(float);
  }
  return emel::kernel::detail::quantized_row_storage_bytes(dtype, static_cast<uint64_t>(cols));
}

inline bool bind_tensor_rows(const tensor_record & tensor,
                             tensor_matrix & out) noexcept {
  out = {};
  if (tensor.data == nullptr || tensor.n_dims <= 0 || tensor.dims[0] <= 0) {
    return false;
  }

  const int32_t cols = static_cast<int32_t>(tensor.dims[0]);
  const int32_t rows = tensor.n_dims > 1 ? static_cast<int32_t>(tensor.dims[1]) : 1;
  if (cols <= 0 || rows <= 0) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(tensor.type);
  if (emel::kernel::detail::is_quantized_k_dtype(dtype) &&
      (cols % static_cast<int32_t>(quant::QK_K)) != 0) {
    return false;
  }

  const size_t row_bytes = row_storage_bytes(tensor, cols);
  if (row_bytes == 0u) {
    return false;
  }

  out.tensor = &tensor;
  out.rows = rows;
  out.cols = cols;
  return true;
}

inline bool copy_tensor_row(const tensor_record & tensor,
                            const int32_t row,
                            std::span<float> out) noexcept {
  if (row < 0 || tensor.data == nullptr || tensor.n_dims <= 0 || tensor.dims[0] <= 0) {
    return false;
  }

  const int32_t cols = static_cast<int32_t>(tensor.dims[0]);
  const int32_t rows = tensor.n_dims > 1 ? static_cast<int32_t>(tensor.dims[1]) : 1;
  const uint8_t dtype = static_cast<uint8_t>(tensor.type);
  if (cols <= 0 ||
      rows <= 0 ||
      row >= rows ||
      static_cast<size_t>(cols) != out.size()) {
    return false;
  }

  const size_t row_bytes = row_storage_bytes(tensor, cols);
  if (row_bytes == 0u) {
    return false;
  }

  const auto * src = static_cast<const uint8_t *>(tensor.data);
  const auto * src_row = src + (static_cast<size_t>(row) * row_bytes);
  switch (static_cast<emel::kernel::event::dtype>(dtype)) {
    case emel::kernel::event::dtype::f32:
      std::memcpy(out.data(), src_row, static_cast<size_t>(cols) * sizeof(float));
      return true;
    case emel::kernel::event::dtype::q2_k:
      quant::dequantize_row_q2_k(reinterpret_cast<const quant::block_q2_k *>(src_row),
                                 out.data(),
                                 cols);
      return true;
    case emel::kernel::event::dtype::q3_k:
      quant::dequantize_row_q3_k(reinterpret_cast<const quant::block_q3_k *>(src_row),
                                 out.data(),
                                 cols);
      return true;
    case emel::kernel::event::dtype::q6_k:
      quant::dequantize_row_q6_k(reinterpret_cast<const quant::block_q6_k *>(src_row),
                                 out.data(),
                                 cols);
      return true;
    default:
      return false;
  }
}

inline bool dequantize_tensor_vector(const tensor_record & tensor,
                                     std::vector<float> & out) noexcept {
  const int32_t cols = tensor.n_dims > 0 ? static_cast<int32_t>(tensor.dims[0]) : 0;
  const int32_t rows = tensor.n_dims > 1 ? static_cast<int32_t>(tensor.dims[1]) : 1;
  if (cols <= 0 || rows != 1) {
    return false;
  }
  out.resize(static_cast<size_t>(cols));
  return copy_tensor_row(tensor, 0, out);
}

inline emel::kernel::event::tensor_view make_src_view(const tensor_matrix & matrix) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  const size_t row_bytes = row_storage_bytes(*matrix.tensor, matrix.cols);

  tensor.data = matrix.tensor->data;
  tensor.type = static_cast<emel::kernel::event::dtype>(matrix.tensor->type);
  tensor.ne = {static_cast<uint64_t>(matrix.cols), static_cast<uint64_t>(matrix.rows), 1u, 1u};
  tensor.nb[0] = dtype == emel::kernel::detail::dtype_f32 ? sizeof(float) : 1u;
  tensor.nb[1] = row_bytes;
  tensor.nb[2] = row_bytes * static_cast<size_t>(matrix.rows);
  tensor.nb[3] = tensor.nb[2];
  return tensor;
}

inline uint64_t matrix_buffer_bytes(const tensor_matrix & matrix) noexcept {
  return static_cast<uint64_t>(row_storage_bytes(*matrix.tensor, matrix.cols)) *
         static_cast<uint64_t>(matrix.rows);
}

inline int32_t append_lifecycle_tensor(
    native_backend & backend,
    void * buffer,
    const uint64_t buffer_bytes,
    const int32_t consumer_refs,
    const bool is_leaf) {
  const int32_t tensor_id = static_cast<int32_t>(backend.lifecycle_tensors.size());
  backend.lifecycle_tensors.push_back(emel::graph::processor::event::lifecycle_tensor_binding{
    .tensor_id = tensor_id,
    .buffer = buffer,
    .buffer_bytes = buffer_bytes,
    .consumer_refs = consumer_refs,
    .is_leaf = is_leaf,
  });
  return tensor_id;
}

inline void append_leaf_lifecycle_tensor(native_backend & backend,
                                         void * buffer,
                                         const uint64_t buffer_bytes) {
  const int32_t tensor_id = append_lifecycle_tensor(backend, buffer, buffer_bytes, 0, true);
  backend.prefill_required_ids.push_back(tensor_id);
  backend.decode_required_ids.push_back(tensor_id);
}

inline void rebuild_lifecycle_views(native_backend & backend) noexcept {
  const auto * tensors = backend.lifecycle_tensors.data();
  backend.prefill_lifecycle_phase = emel::graph::processor::event::lifecycle_phase{
    .required_filled_ids = backend.prefill_required_ids.data(),
    .required_filled_count = static_cast<int32_t>(backend.prefill_required_ids.size()),
    .publish_ids = backend.prefill_publish_ids.data(),
    .publish_count = static_cast<int32_t>(backend.prefill_publish_ids.size()),
    .release_ids = backend.prefill_release_ids.data(),
    .release_count = static_cast<int32_t>(backend.prefill_release_ids.size()),
  };
  backend.decode_lifecycle_phase = emel::graph::processor::event::lifecycle_phase{
    .required_filled_ids = backend.decode_required_ids.data(),
    .required_filled_count = static_cast<int32_t>(backend.decode_required_ids.size()),
    .publish_ids = backend.decode_publish_ids.data(),
    .publish_count = static_cast<int32_t>(backend.decode_publish_ids.size()),
    .release_ids = backend.decode_release_ids.data(),
    .release_count = static_cast<int32_t>(backend.decode_release_ids.size()),
  };
  backend.reserve_lifecycle = emel::graph::processor::event::lifecycle_manifest{
    .tensors = tensors,
    .tensor_count = static_cast<int32_t>(backend.lifecycle_tensors.size()),
    .phase = nullptr,
  };
  backend.prefill_lifecycle = emel::graph::processor::event::lifecycle_manifest{
    .tensors = tensors,
    .tensor_count = static_cast<int32_t>(backend.lifecycle_tensors.size()),
    .phase = &backend.prefill_lifecycle_phase,
  };
  backend.decode_lifecycle = emel::graph::processor::event::lifecycle_manifest{
    .tensors = tensors,
    .tensor_count = static_cast<int32_t>(backend.lifecycle_tensors.size()),
    .phase = &backend.decode_lifecycle_phase,
  };
}

inline void build_lifecycle(native_backend & backend) {
  backend.lifecycle_tensors.clear();
  backend.prefill_required_ids.clear();
  backend.prefill_publish_ids.clear();
  backend.prefill_release_ids.clear();
  backend.decode_required_ids.clear();
  backend.decode_publish_ids.clear();
  backend.decode_release_ids.clear();

  append_leaf_lifecycle_tensor(
      backend,
      const_cast<void *>(backend.token_embedding.tensor->data),
      matrix_buffer_bytes(backend.token_embedding));
  append_leaf_lifecycle_tensor(
      backend,
      backend.output_norm.data(),
      static_cast<uint64_t>(backend.output_norm.size()) * sizeof(float));
  append_leaf_lifecycle_tensor(
      backend,
      const_cast<void *>(backend.output.tensor->data),
      matrix_buffer_bytes(backend.output));

  for (auto & block : backend.blocks) {
    append_leaf_lifecycle_tensor(
        backend,
        block.attention_norm.data(),
        static_cast<uint64_t>(block.attention_norm.size()) * sizeof(float));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.attention_q.tensor->data),
        matrix_buffer_bytes(block.attention_q));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.attention_k.tensor->data),
        matrix_buffer_bytes(block.attention_k));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.attention_v.tensor->data),
        matrix_buffer_bytes(block.attention_v));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.attention_output.tensor->data),
        matrix_buffer_bytes(block.attention_output));
    append_leaf_lifecycle_tensor(
        backend,
        block.feed_forward_norm.data(),
        static_cast<uint64_t>(block.feed_forward_norm.size()) * sizeof(float));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.feed_forward_gate.tensor->data),
        matrix_buffer_bytes(block.feed_forward_gate));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.feed_forward_down.tensor->data),
        matrix_buffer_bytes(block.feed_forward_down));
    append_leaf_lifecycle_tensor(
        backend,
        const_cast<void *>(block.feed_forward_up.tensor->data),
        matrix_buffer_bytes(block.feed_forward_up));
  }

  backend.input_tokens_tensor_id = append_lifecycle_tensor(backend, nullptr, 0u, 0, true);
  backend.positions_tensor_id = append_lifecycle_tensor(backend, nullptr, 0u, 0, true);
  backend.logits_tensor_id = append_lifecycle_tensor(backend, nullptr, 0u, 1, false);
  backend.key_cache_tensor_id = append_lifecycle_tensor(
      backend,
      backend.key_cache.data(),
      static_cast<uint64_t>(backend.key_cache.size()) * sizeof(float),
      1,
      false);
  backend.value_cache_tensor_id = append_lifecycle_tensor(
      backend,
      backend.value_cache.data(),
      static_cast<uint64_t>(backend.value_cache.size()) * sizeof(float),
      1,
      false);

  backend.prefill_required_ids.push_back(backend.input_tokens_tensor_id);
  backend.prefill_required_ids.push_back(backend.positions_tensor_id);
  backend.prefill_publish_ids.push_back(backend.logits_tensor_id);
  backend.prefill_publish_ids.push_back(backend.key_cache_tensor_id);
  backend.prefill_publish_ids.push_back(backend.value_cache_tensor_id);
  backend.prefill_release_ids.push_back(backend.logits_tensor_id);

  backend.decode_required_ids = backend.prefill_required_ids;
  backend.decode_required_ids.push_back(backend.key_cache_tensor_id);
  backend.decode_required_ids.push_back(backend.value_cache_tensor_id);
  backend.decode_publish_ids.push_back(backend.logits_tensor_id);
  backend.decode_release_ids = backend.prefill_release_ids;

  rebuild_lifecycle_views(backend);
}

inline void bind_runtime_lifecycle(native_backend & backend,
                                   int32_t * input_tokens,
                                   const int32_t input_token_capacity,
                                   int32_t * positions,
                                   const int32_t position_capacity,
                                   float * logits,
                                   const int32_t logits_capacity) noexcept {
  backend.lifecycle_tensors[static_cast<size_t>(backend.input_tokens_tensor_id)].buffer =
      input_tokens;
  backend.lifecycle_tensors[static_cast<size_t>(backend.input_tokens_tensor_id)].buffer_bytes =
      static_cast<uint64_t>(input_token_capacity) * sizeof(int32_t);
  backend.lifecycle_tensors[static_cast<size_t>(backend.positions_tensor_id)].buffer = positions;
  backend.lifecycle_tensors[static_cast<size_t>(backend.positions_tensor_id)].buffer_bytes =
      static_cast<uint64_t>(position_capacity) * sizeof(int32_t);
  backend.lifecycle_tensors[static_cast<size_t>(backend.logits_tensor_id)].buffer = logits;
  backend.lifecycle_tensors[static_cast<size_t>(backend.logits_tensor_id)].buffer_bytes =
      static_cast<uint64_t>(logits_capacity) * sizeof(float);
}

inline const emel::graph::processor::event::lifecycle_manifest * reserve_lifecycle(
    native_backend & backend,
    int32_t * input_tokens,
    const int32_t input_token_capacity,
    int32_t * positions,
    const int32_t position_capacity,
    float * logits,
    const int32_t logits_capacity) noexcept {
  bind_runtime_lifecycle(
      backend, input_tokens, input_token_capacity, positions, position_capacity, logits,
      logits_capacity);
  return &backend.reserve_lifecycle;
}

inline const emel::graph::processor::event::lifecycle_manifest * phase_lifecycle(
    native_backend & backend,
    int32_t * input_tokens,
    const int32_t input_token_capacity,
    int32_t * positions,
    const int32_t position_capacity,
    float * logits,
    const int32_t logits_capacity,
    const step_kind kind) noexcept {
  bind_runtime_lifecycle(
      backend, input_tokens, input_token_capacity, positions, position_capacity, logits,
      logits_capacity);
  const std::array<const emel::graph::processor::event::lifecycle_manifest *, 2> manifests{
    &backend.prefill_lifecycle,
    &backend.decode_lifecycle,
  };
  return manifests[static_cast<size_t>(kind)];
}

inline bool matmul_vector(native_backend & backend,
                          const tensor_matrix & matrix,
                          std::span<const float> input,
                          std::span<float> output) noexcept {
  if (matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_src_view(
          input.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(input.size())),
      .dst = make_dst_view(
          output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
      .nth = 1,
  };
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(ev);
  backend.kernel_dispatch_calls += 1;
  return ok;
}

inline bool rms_norm(std::span<const float> input,
                     std::span<const float> weight,
                     const float epsilon,
                     std::span<float> output) noexcept {
  if (input.size() != weight.size() || input.size() != output.size() || input.empty()) {
    return false;
  }

  double square_sum = 0.0;
  for (const float value : input) {
    square_sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const float mean = static_cast<float>(square_sum / static_cast<double>(input.size()));
  const float scale = 1.0f / std::sqrt(mean + epsilon);
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i];
    output[i] *= scale;
    output[i] *= weight[i];
  }
  return true;
}

inline void apply_rope(std::span<float> vector,
                       const int32_t head_count,
                       const int32_t head_dim,
                       const int32_t n_rot,
                       const int32_t position,
                       const float rope_freq_base) noexcept {
  const int32_t rot_dim = std::min(n_rot, head_dim);
  if (head_count <= 0 || head_dim <= 1 || rot_dim <= 1) {
    return;
  }

  for (int32_t head = 0; head < head_count; ++head) {
    float * head_ptr =
        vector.data() + (static_cast<size_t>(head) * static_cast<size_t>(head_dim));
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

inline float silu(const float value) noexcept {
  return value / (1.0f + ::expf(-value));
}

inline size_t layer_cache_offset(const native_backend & backend,
                                 const int32_t layer,
                                 const int32_t position,
                                 const int32_t kv_dim) noexcept {
  return ((static_cast<size_t>(layer) * static_cast<size_t>(backend.n_ctx)) +
          static_cast<size_t>(position)) *
         static_cast<size_t>(kv_dim);
}

inline void store_fp16_rounded_cache(std::span<const float> src, float * dst) noexcept {
  for (size_t idx = 0; idx < src.size(); ++idx) {
    dst[idx] = quant::fp16_to_fp32(quant::fp32_to_fp16(src[idx]));
  }
}

inline bool check_backend(const native_backend * backend, int32_t * err_out) noexcept {
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

inline const step_plan * request_plan(const emel::graph::processor::event::execute & request,
                                      int32_t * err_out) noexcept {
  const auto * plan = static_cast<const step_plan *>(request.step_plan);
  if (plan == nullptr || plan->graph == nullptr || plan->graph->execution == nullptr) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return nullptr;
  }
  return plan;
}

inline bool store_bound_request(native_backend & backend,
                                const emel::graph::processor::event::execute & request,
                                int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  if (io == nullptr ||
      io->token_ids == nullptr ||
      io->token_count <= 0 ||
      static_cast<size_t>(io->token_count) > backend.bound_tokens.size() ||
      request.positions == nullptr ||
      request.positions_count != io->token_count ||
      static_cast<size_t>(request.positions_count) > backend.bound_positions.size()) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  std::copy_n(io->token_ids, io->token_count, backend.bound_tokens.begin());
  std::copy_n(request.positions, request.positions_count, backend.bound_positions.begin());
  backend.bound_token_count = io->token_count;
  backend.bound_position_count = request.positions_count;
  backend.bound_ready = true;
  return true;
}

inline bool compute_attention(native_backend & backend,
                              const int32_t layer_index,
                              const int32_t position_limit,
                              const std::span<const float> q_vector) noexcept {
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
      const size_t cache_offset =
          layer_cache_offset(backend, layer_index, position, kv_dim) + kv_offset;
      const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
          q_vector.data() + static_cast<std::ptrdiff_t>(q_offset),
          backend.key_cache.data() + static_cast<std::ptrdiff_t>(cache_offset),
          static_cast<uint64_t>(head_dim)) *
          inv_scale;
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
      const size_t cache_offset =
          layer_cache_offset(backend, layer_index, position, kv_dim) + kv_offset;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        backend.attn_ctx[q_offset + static_cast<size_t>(dim)] +=
            weight * backend.value_cache[cache_offset + static_cast<size_t>(dim)];
      }
    }
  }

  return true;
}

inline emel::kernel::event::op_flash_attn_ext make_flash_attn_request(
    native_backend & backend,
    const int32_t layer_index,
    const int32_t position) noexcept {
  emel::kernel::event::op_flash_attn_ext request{};
  const uint64_t kv_tokens = static_cast<uint64_t>(position + 1);
  const uint64_t head_dim = static_cast<uint64_t>(backend.head_dim);
  const uint64_t head_count = static_cast<uint64_t>(backend.n_head);
  const uint64_t kv_head_dim = static_cast<uint64_t>(backend.head_dim_kv);
  const uint64_t kv_head_count = static_cast<uint64_t>(backend.n_head_kv);
  const uint64_t kv_dim = kv_head_dim * kv_head_count;
  const size_t layer_offset = layer_cache_offset(
      backend, layer_index, 0, static_cast<int32_t>(kv_dim));
  const float scale = 1.0f / std::sqrt(static_cast<float>(backend.head_dim));

  request.src0 = make_src_view_3d(backend.q_attn.data(), head_dim, 1u, head_count);
  request.src1 = make_src_view_strided_3d(backend.key_cache.data() + layer_offset,
                                          kv_head_dim,
                                          kv_tokens,
                                          kv_head_count,
                                          sizeof(float) * kv_dim,
                                          sizeof(float) * kv_head_dim);
  request.src2 = make_src_view_strided_3d(backend.value_cache.data() + layer_offset,
                                          kv_head_dim,
                                          kv_tokens,
                                          kv_head_count,
                                          sizeof(float) * kv_dim,
                                          sizeof(float) * kv_head_dim);
  request.dst = make_dst_view_3d(backend.attn_ctx.data(), head_dim, 1u, head_count);
  request.nth = 1;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  request.op_params_size = sizeof(scale);
  return request;
}

inline bool dispatch_flash_attention(native_backend & backend,
                                     const int32_t layer_index,
                                     const int32_t position) noexcept {
  const auto request = make_flash_attn_request(backend, layer_index, position);
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(request);
  ++backend.kernel_dispatch_calls;
  backend.flash_attention_dispatch_calls += static_cast<uint64_t>(ok);
  return ok;
}

inline bool run_layer(native_backend & backend,
                      const int32_t layer_index,
                      const int32_t position) noexcept {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!rms_norm(backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
      !matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
      !matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
    return false;
  }

  apply_rope(
      backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
  apply_rope(backend.k,
             backend.n_head_kv,
             backend.head_dim_kv,
             backend.n_rot,
             position,
             backend.rope_freq_base);
  store_fp16_rounded_cache(
      std::span<const float>(backend.q.data(), static_cast<size_t>(backend.n_embd)),
      backend.q_attn.data());

  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const size_t cache_offset = layer_cache_offset(backend, layer_index, position, kv_dim);
  store_fp16_rounded_cache(
      std::span<const float>(backend.k.data(), static_cast<size_t>(kv_dim)),
      backend.key_cache.data() + cache_offset);
  store_fp16_rounded_cache(
      std::span<const float>(backend.v.data(), static_cast<size_t>(kv_dim)),
      backend.value_cache.data() + cache_offset);

  if (!dispatch_flash_attention(backend, layer_index, position) ||
      !matmul_vector(backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!rms_norm(backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm) ||
      !matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
      !matmul_vector(backend, block.feed_forward_up, backend.norm, backend.up)) {
    return false;
  }

  for (size_t idx = 0; idx < backend.gate.size(); ++idx) {
    backend.ffn_hidden[idx] = silu(backend.gate[idx]) * backend.up[idx];
  }

  if (!matmul_vector(backend, block.feed_forward_down, backend.ffn_hidden, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

inline bool compute_logits(native_backend & backend) noexcept {
  return rms_norm(backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm) &&
         matmul_vector(backend, backend.output, backend.norm, backend.bound_logits);
}

inline bool run_prefill(native_backend & backend) noexcept {
  backend.kv_cache_tokens = 0;

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  for (size_t token_index = 0; token_index < token_count; ++token_index) {
    const int32_t token_id = backend.bound_tokens[token_index];
    const int32_t position = backend.bound_positions[token_index];
    if (token_id < 0 ||
        token_id >= backend.token_embedding.rows ||
        position < 0 ||
        position >= backend.n_ctx) {
      return false;
    }

    if (!copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
      return false;
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return compute_logits(backend);
}

inline bool run_decode(native_backend & backend,
                       const emel::graph::processor::event::execute & request) noexcept {
  if (backend.bound_token_count != 1 ||
      backend.bound_position_count != 1 ||
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

  if (!copy_tensor_row(*backend.token_embedding.tensor, token_id, backend.hidden)) {
    return false;
  }

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!run_layer(backend, layer, position)) {
      return false;
    }
  }
  backend.kv_cache_tokens = position + 1;
  return compute_logits(backend);
}

}  // namespace

inline emel::error::type prepare(native_backend & backend,
                                 const emel::model::data & model_data) noexcept {
  std::destroy_at(std::addressof(backend));
  std::construct_at(std::addressof(backend));
  backend.kernel_kind = detect_host_kernel_kind();
  backend.kernel.set_kind(backend.kernel_kind);

  if (emel::model::llama::detail::build_execution_view(model_data, backend.execution) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::model::llama::detail::build_topology(backend.execution, backend.topology) !=
          emel::error::cast(emel::model::loader::error::none) ||
      emel::model::llama::detail::build_step_plans(
          backend.topology, backend.prefill_plan, backend.decode_plan) !=
          emel::error::cast(emel::model::loader::error::none)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  backend.model = &model_data;
  backend.n_vocab = model_data.params.n_vocab;
  backend.n_embd = model_data.params.n_embd;
  backend.n_head = model_data.params.n_head;
  backend.n_head_kv =
      model_data.params.n_head_kv > 0 ? model_data.params.n_head_kv : model_data.params.n_head;
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
      (backend.n_embd % backend.n_head) != 0) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  backend.head_dim = backend.n_embd / backend.n_head;

  if (!bind_tensor_rows(*backend.execution.token_embedding.tensor, backend.token_embedding) ||
      !dequantize_tensor_vector(*backend.execution.output_norm.tensor, backend.output_norm) ||
      !bind_tensor_rows(*backend.execution.output.tensor, backend.output)) {
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
        !bind_tensor_rows(*block.attention_q.tensor, weights.attention_q) ||
        !bind_tensor_rows(*block.attention_k.tensor, weights.attention_k) ||
        !bind_tensor_rows(*block.attention_v.tensor, weights.attention_v) ||
        !bind_tensor_rows(*block.attention_output.tensor, weights.attention_output) ||
        !dequantize_tensor_vector(*block.feed_forward_norm.tensor, weights.feed_forward_norm) ||
        !bind_tensor_rows(*block.feed_forward_gate.tensor, weights.feed_forward_gate) ||
        !bind_tensor_rows(*block.feed_forward_down.tensor, weights.feed_forward_down) ||
        !bind_tensor_rows(*block.feed_forward_up.tensor, weights.feed_forward_up)) {
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
  backend.bound_tokens.resize(static_cast<size_t>(backend.n_ctx));
  backend.bound_positions.resize(static_cast<size_t>(backend.n_ctx));
  backend.hidden.resize(static_cast<size_t>(backend.n_embd));
  backend.norm.resize(static_cast<size_t>(backend.n_embd));
  backend.q.resize(static_cast<size_t>(backend.n_embd));
  backend.q_attn.resize(static_cast<size_t>(backend.n_embd));
  backend.k.resize(static_cast<size_t>(kv_dim));
  backend.v.resize(static_cast<size_t>(kv_dim));
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(static_cast<size_t>(backend.n_embd));
  backend.projected.resize(static_cast<size_t>(backend.n_embd));
  backend.gate.resize(static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));
  backend.up.resize(static_cast<size_t>(backend.blocks[0].feed_forward_up.rows));
  backend.ffn_hidden.resize(static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));
  build_lifecycle(backend);

  return emel::error::cast(emel::model::loader::error::none);
}

inline bool validate(const emel::graph::processor::event::execute & request,
                     int32_t * err_out) noexcept {
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

inline bool prepare_graph(const emel::graph::processor::event::execute &,
                          bool * reused_out,
                          int32_t * err_out) noexcept {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = k_error_ok;
  }
  return true;
}

inline bool alloc_graph(const emel::graph::processor::event::execute &,
                        int32_t * err_out) noexcept {
  if (err_out != nullptr) {
    *err_out = k_error_ok;
  }
  return true;
}

inline bool bind_inputs(const emel::graph::processor::event::execute & request,
                        int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  if (!check_backend(backend, err_out)) {
    return false;
  }
  return store_bound_request(*backend, request, err_out);
}

inline bool run_kernel(const emel::graph::processor::event::execute & request,
                       int32_t * err_out) noexcept {
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

inline bool extract_outputs(const emel::graph::processor::event::execute & request,
                            int32_t * outputs_out,
                            int32_t * err_out) noexcept {
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

  std::copy(backend->bound_logits.begin(), backend->bound_logits.end(), io->logits);
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

}  // namespace emel::generator::detail
