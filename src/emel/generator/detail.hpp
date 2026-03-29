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

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

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

struct packed_q8_0_binding {
  emel::model::data::tensor_record tensor = {};
  std::vector<uint8_t> storage = {};
};

struct block_weights {
  std::vector<float> attention_norm = {};
  tensor_matrix attention_q = {};
  packed_q8_0_binding attention_q_packed = {};
  tensor_matrix attention_k = {};
  packed_q8_0_binding attention_k_packed = {};
  tensor_matrix attention_v = {};
  packed_q8_0_binding attention_v_packed = {};
  std::vector<float> attention_q_norm = {};
  std::vector<float> attention_k_norm = {};
  tensor_matrix attention_output = {};
  packed_q8_0_binding attention_output_packed = {};
  std::vector<float> feed_forward_norm = {};
  tensor_matrix feed_forward_gate = {};
  packed_q8_0_binding feed_forward_gate_packed = {};
  tensor_matrix feed_forward_down = {};
  packed_q8_0_binding feed_forward_down_packed = {};
  tensor_matrix feed_forward_up = {};
  packed_q8_0_binding feed_forward_up_packed = {};
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
  uint64_t native_q8_0_dispatch_calls = 0;
  uint64_t packed_q8_0_dispatch_calls = 0;
  uint64_t flash_attention_dispatch_calls = 0;

  tensor_matrix token_embedding = {};
  std::vector<float> output_norm = {};
  tensor_matrix output_native = {};
  tensor_matrix output = {};
  tensor_matrix output_argmax = {};
  emel::model::data::tensor_record output_packed_tensor = {};
  std::vector<uint8_t> output_packed_storage = {};
  emel::model::data::tensor_record output_prepared_tensor = {};
  std::vector<uint8_t> output_prepared_storage = {};
  emel::model::data::tensor_record output_argmax_packed_tensor = {};
  std::vector<uint8_t> output_argmax_packed_storage = {};
  emel::model::data::tensor_record output_argmax_prepared_tensor = {};
  std::vector<uint8_t> output_argmax_prepared_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_k> logits_input_q8_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_0> packed_q8_0_input_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_0> packed_q8_0_chunk4_rows = {};
  std::vector<uint8_t> packed_q8_0_chunk4_input_storage = {};
  emel::model::llama::detail::quantized_path_audit quantized_audit = {};
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

  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<uint16_t> flash_key_cache = {};
  std::vector<uint16_t> flash_value_cache = {};
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
  std::vector<float> hidden_chunk4 = {};
  std::vector<float> norm = {};
  std::vector<float> norm_chunk4 = {};
  std::vector<float> q = {};
  std::vector<float> q_attn = {};
  std::vector<float> q_chunk4 = {};
  std::vector<float> k = {};
  std::vector<float> k_chunk4 = {};
  std::vector<float> v = {};
  std::vector<float> v_chunk4 = {};
  std::vector<float> attn_scores = {};
  std::vector<float> attn_probs = {};
  std::vector<float> attn_probs_rounded = {};
  std::vector<float> attn_value_column = {};
  std::vector<float> attn_ctx = {};
  std::vector<float> attn_ctx_chunk4 = {};
  std::vector<float> projected = {};
  std::vector<float> projected_chunk4 = {};
  std::vector<float> gate = {};
  std::vector<float> gate_chunk4 = {};
  std::vector<float> up = {};
  std::vector<float> up_chunk4 = {};
  std::vector<float> ffn_hidden = {};
  std::vector<float> ffn_hidden_chunk4 = {};
  bool bound_ready = false;
};

namespace quant = emel::kernel::detail::quant;

namespace {

using tensor_record = emel::model::data::tensor_record;
using step_kind = emel::model::llama::detail::step_kind;
using step_plan = emel::model::llama::detail::step_plan;

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_invalid = 1;
constexpr int32_t k_prefill_q8_chunk_rows = 4;
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
  const uint64_t elem_size =
      emel::kernel::detail::dtype_size_bytes(emel::kernel::detail::dtype_code(tensor.type));
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

inline emel::kernel::event::tensor_view make_src_view(const void * data,
                                                      const emel::kernel::event::dtype type,
                                                      const uint64_t ne0,
                                                      const uint64_t ne1 = 1u) noexcept {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = type;
  tensor.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(tensor);
  return tensor;
}

inline emel::kernel::event::tensor_view make_src_view(const float * data,
                                                      const uint64_t ne0,
                                                      const uint64_t ne1 = 1u) noexcept {
  return make_src_view(data, emel::kernel::event::dtype::f32, ne0, ne1);
}

inline emel::kernel::event::tensor_view make_q8_k_vector_view(
    const emel::kernel::detail::quant::block_q8_k * data,
    const uint64_t cols) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const size_t row_bytes =
      emel::kernel::detail::quantized_row_storage_bytes(
          emel::kernel::detail::dtype_q8_k, cols);
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::q8_k;
  tensor.ne = {1u, cols, 1u, 1u};
  tensor.nb[0] = 1u;
  tensor.nb[1] = row_bytes;
  tensor.nb[2] = row_bytes;
  tensor.nb[3] = row_bytes;
  return tensor;
}

inline emel::kernel::event::tensor_view make_q8_0_vector_view(
    const emel::kernel::detail::quant::block_q8_0 * data,
    const uint64_t cols) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const size_t row_bytes =
      emel::kernel::detail::quantized_row_storage_bytes(
          emel::kernel::detail::dtype_q8_0, cols);
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::q8_0;
  tensor.ne = {1u, cols, 1u, 1u};
  tensor.nb[0] = 1u;
  tensor.nb[1] = row_bytes;
  tensor.nb[2] = row_bytes;
  tensor.nb[3] = row_bytes;
  return tensor;
}

inline emel::kernel::event::tensor_view make_packed_q8_0_rhs_chunk4_view(
    const void * data,
    const uint64_t cols) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const size_t group_bytes =
      emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(cols);
  const uint64_t group_count =
      emel::kernel::detail::quant::packed_q8_0_x4_group_count(k_prefill_q8_chunk_rows);
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::q8_0_x4_bl8;
  tensor.ne = {
      static_cast<uint64_t>(k_prefill_q8_chunk_rows),
      cols,
      1u,
      1u,
  };
  tensor.nb[0] = 1u;
  tensor.nb[1] = group_bytes;
  tensor.nb[2] = group_bytes * group_count;
  tensor.nb[3] = tensor.nb[2];
  return tensor;
}

inline emel::kernel::event::tensor_view make_src_view_3d(const void * data,
                                                         const emel::kernel::event::dtype type,
                                                         const uint64_t ne0,
                                                         const uint64_t ne1,
                                                         const uint64_t ne2) noexcept {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = type;
  tensor.ne = {ne0, ne1, ne2, 1u};
  fill_default_nb(tensor);
  return tensor;
}

inline emel::kernel::event::tensor_view make_src_view_3d(const float * data,
                                                         const uint64_t ne0,
                                                         const uint64_t ne1,
                                                         const uint64_t ne2) noexcept {
  return make_src_view_3d(data, emel::kernel::event::dtype::f32, ne0, ne1, ne2);
}

inline emel::kernel::event::tensor_view make_src_view_strided_3d(const void * data,
                                                                 const emel::kernel::event::dtype type,
                                                                 const uint64_t ne0,
                                                                 const uint64_t ne1,
                                                                 const uint64_t ne2,
                                                                 const uint64_t nb1,
                                                                 const uint64_t nb2) noexcept {
  emel::kernel::event::tensor_view tensor{};
  tensor.data = data;
  tensor.type = type;
  tensor.ne = {ne0, ne1, ne2, 1u};
  tensor.nb[0] =
      emel::kernel::detail::dtype_size_bytes(emel::kernel::detail::dtype_code(type));
  tensor.nb[1] = nb1;
  tensor.nb[2] = nb2;
  tensor.nb[3] = nb2 * ne2;
  return tensor;
}

inline emel::kernel::event::tensor_view make_src_view_strided_3d(const float * data,
                                                                 const uint64_t ne0,
                                                                 const uint64_t ne1,
                                                                 const uint64_t ne2,
                                                                 const uint64_t nb1,
                                                                 const uint64_t nb2) noexcept {
  return make_src_view_strided_3d(
      data, emel::kernel::event::dtype::f32, ne0, ne1, ne2, nb1, nb2);
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

inline emel::kernel::event::tensor_view_mut make_batch_major_dst_view(
    float * data,
    const uint64_t rows,
    const uint64_t cols) noexcept {
  emel::kernel::event::tensor_view_mut tensor{};
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::f32;
  tensor.ne = {rows, cols, 1u, 1u};
  tensor.nb[0] = sizeof(float) * cols;
  tensor.nb[1] = sizeof(float);
  tensor.nb[2] = tensor.nb[0] * rows;
  tensor.nb[3] = tensor.nb[2];
  return tensor;
}

inline size_t row_storage_bytes(const tensor_record & tensor, const int32_t cols) noexcept {
  const uint8_t dtype = static_cast<uint8_t>(tensor.type);
  if (dtype == emel::kernel::detail::dtype_f32) {
    return static_cast<size_t>(cols) * sizeof(float);
  }
  if (dtype == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
      dtype == emel::kernel::detail::dtype_q8_0_x4_bl8) {
    return emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(
        static_cast<uint64_t>(cols));
  }
  if (dtype == emel::kernel::detail::dtype_q6_k_x8) {
    return emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(
        static_cast<uint64_t>(cols));
  }
  if (dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared) {
    return emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(
        static_cast<uint64_t>(cols));
  }
  if (dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared) {
    return emel::kernel::detail::quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(
        static_cast<uint64_t>(cols));
  }
  return emel::kernel::detail::quantized_row_storage_bytes(dtype, static_cast<uint64_t>(cols));
}

inline bool packed_q6_k_x8_logits_supported(const native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return backend.kernel_kind == emel::kernel::kernel_kind::aarch64;
#else
  (void) backend;
  return false;
#endif
}

inline bool prepared_q6_k_x8_q8_logits_supported(const native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  return backend.kernel_kind == emel::kernel::kernel_kind::aarch64;
#else
  (void) backend;
  return false;
#endif
}

inline bool packed_q8_0_input_path_supported(const native_backend & backend,
                                             const tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.packed_q8_0_input_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  return emel::kernel::detail::is_packed_q8_0_vector_dtype(
      static_cast<uint8_t>(matrix.tensor->type));
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool packed_q8_0_chunk4_input_path_supported(const native_backend & backend,
                                                    const tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.packed_q8_0_chunk4_rows.empty() ||
      backend.packed_q8_0_chunk4_input_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  return static_cast<uint8_t>(matrix.tensor->type) == emel::kernel::detail::dtype_q8_0_x4_bl8 &&
      (matrix.rows % k_prefill_q8_chunk_rows) == 0;
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool prefill_chunk4_q8_gemm_supported(const native_backend & backend) noexcept {
  if (backend.blocks.empty() || backend.n_layer <= 0) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    if (!packed_q8_0_chunk4_input_path_supported(backend, block.attention_q) ||
        !packed_q8_0_chunk4_input_path_supported(backend, block.attention_k) ||
        !packed_q8_0_chunk4_input_path_supported(backend, block.attention_v) ||
        !packed_q8_0_chunk4_input_path_supported(backend, block.attention_output) ||
        !packed_q8_0_chunk4_input_path_supported(backend, block.feed_forward_gate) ||
        !packed_q8_0_chunk4_input_path_supported(backend, block.feed_forward_down) ||
        !packed_q8_0_chunk4_input_path_supported(backend, block.feed_forward_up)) {
      return false;
    }
  }

  return backend.hidden_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(backend.n_embd) &&
      backend.norm_chunk4.size() == backend.hidden_chunk4.size() &&
      backend.projected_chunk4.size() == backend.hidden_chunk4.size() &&
      backend.attn_ctx_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(backend.n_head * backend.head_dim) &&
      backend.q_chunk4.size() == backend.attn_ctx_chunk4.size() &&
      backend.k_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(backend.n_head_kv * backend.head_dim_kv) &&
      backend.v_chunk4.size() == backend.k_chunk4.size() &&
      backend.gate_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(backend.blocks.front().feed_forward_gate.rows) &&
      backend.up_chunk4.size() == backend.gate_chunk4.size() &&
      backend.ffn_hidden_chunk4.size() == backend.gate_chunk4.size();
}

inline bool q8_logits_input_path_supported(const tensor_matrix & matrix) noexcept {
  if (matrix.tensor == nullptr) {
    return false;
  }
  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  return dtype == emel::kernel::detail::dtype_q6_k_x8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
}

inline bool preselected_argmax_direct_supported(const native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.output_argmax.tensor == nullptr ||
      backend.logits_input_q8_storage.empty()) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(backend.output_argmax.tensor->type);
#if defined(__ARM_FEATURE_DOTPROD)
  if (dtype == emel::kernel::detail::dtype_q6_k_x8) {
    return true;
  }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared) {
    return true;
  }
#endif
  return false;
#else
  (void) backend;
  return false;
#endif
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
  const uint64_t block_size = emel::kernel::detail::quantized_block_size(dtype);
  if (block_size != 0u && (static_cast<uint64_t>(cols) % block_size) != 0u) {
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
    case emel::kernel::event::dtype::q8_0:
      quant::dequantize_row_q8_0(reinterpret_cast<const quant::block_q8_0 *>(src_row),
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

inline bool is_qwen3_runtime(const native_backend & backend) noexcept {
  return backend.model != nullptr && emel::model::architecture_name_view(*backend.model) == "qwen3";
}

inline const tensor_record * select_output_projection_tensor(
    const emel::model::llama::detail::execution_view & execution) noexcept {
  if (execution.output.tensor != nullptr) {
    return execution.output.tensor;
  }

  const auto * token_embedding = execution.token_embedding.tensor;
  if (token_embedding == nullptr || token_embedding->data == nullptr || token_embedding->n_dims != 2) {
    return nullptr;
  }

  return token_embedding;
}

inline bool bind_output_projection(native_backend & backend) noexcept {
  const auto * tensor = select_output_projection_tensor(backend.execution);
  return tensor != nullptr && bind_tensor_rows(*tensor, backend.output_native);
}

inline bool apply_headwise_rms_norm(std::span<float> vector,
                                    std::span<const float> weights,
                                    const int32_t head_count,
                                    const int32_t head_dim,
                                    const float epsilon) noexcept {
  if (head_count <= 0 ||
      head_dim <= 0 ||
      static_cast<size_t>(head_count) * static_cast<size_t>(head_dim) != vector.size() ||
      static_cast<size_t>(head_dim) != weights.size()) {
    return false;
  }

  for (int32_t head = 0; head < head_count; ++head) {
    const size_t head_offset =
        static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    float square_sum = 0.0f;
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      const float value = vector[head_offset + static_cast<size_t>(dim)];
      square_sum += value * value;
    }

    const float inv_rms =
        1.0f / std::sqrt(square_sum / static_cast<float>(head_dim) + epsilon);
    for (int32_t dim = 0; dim < head_dim; ++dim) {
      vector[head_offset + static_cast<size_t>(dim)] *=
          inv_rms * weights[static_cast<size_t>(dim)];
    }
  }

  return true;
}

inline bool apply_qwen3_attention_qk_norm(native_backend & backend,
                                          const block_weights & block) noexcept {
  return apply_headwise_rms_norm(
             backend.q, block.attention_q_norm, backend.n_head, backend.head_dim, backend.rms_epsilon) &&
      apply_headwise_rms_norm(
             backend.k, block.attention_k_norm, backend.n_head_kv, backend.head_dim_kv, backend.rms_epsilon);
}

inline void reset_output_logits(native_backend & backend) noexcept {
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;
  backend.output_packed_tensor = {};
  backend.output_packed_storage.clear();
  backend.output_prepared_tensor = {};
  backend.output_prepared_storage.clear();
  backend.output_argmax_packed_tensor = {};
  backend.output_argmax_packed_storage.clear();
  backend.output_argmax_prepared_tensor = {};
  backend.output_argmax_prepared_storage.clear();
}

template <uint8_t packed_dtype>
inline bool prepare_packed_q8_0_tensor_layout(const tensor_record & source,
                                              const int32_t rows,
                                              const int32_t cols,
                                              tensor_record & packed_tensor,
                                              std::vector<uint8_t> & packed_storage) noexcept {
  if (source.data == nullptr ||
      static_cast<uint8_t>(source.type) != emel::kernel::detail::dtype_q8_0 ||
      rows <= 0 ||
      cols <= 0) {
    return false;
  }

  const uint64_t urows = static_cast<uint64_t>(rows);
  const uint64_t ucols = static_cast<uint64_t>(cols);
  const uint64_t group_count = emel::kernel::detail::quant::packed_q8_0_x4_group_count(urows);
  const size_t group_bytes =
      emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(ucols);
  const uint64_t storage_bytes = static_cast<uint64_t>(group_bytes) * group_count;
  if (group_bytes == 0u || storage_bytes == 0u) {
    return false;
  }

  packed_storage.resize(static_cast<size_t>(storage_bytes));
  const auto * src =
      reinterpret_cast<const emel::kernel::detail::quant::block_q8_0 *>(source.data);
  bool packed_ok = false;
  if constexpr (packed_dtype == emel::kernel::detail::dtype_q8_0_x4_bl8) {
    packed_ok = emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
        src, urows, ucols, packed_storage.data());
  } else if constexpr (packed_dtype == emel::kernel::detail::dtype_q8_0_x4_bl4) {
    packed_ok = emel::kernel::detail::quant::pack_q8_0_rows_x4_bl4(
        src, urows, ucols, packed_storage.data());
  }
  if (!packed_ok) {
    return false;
  }

  packed_tensor = source;
  packed_tensor.type = packed_dtype;
  packed_tensor.data = packed_storage.data();
  packed_tensor.data_size = storage_bytes;
  return true;
}

template <uint8_t packed_dtype>
inline bool prepare_packed_q8_0_matrix_layout(tensor_matrix & matrix,
                                              packed_q8_0_binding & packed) noexcept {
  if (matrix.tensor == nullptr) {
    return false;
  }
  if (static_cast<uint8_t>(matrix.tensor->type) != emel::kernel::detail::dtype_q8_0) {
    return true;
  }
  if (!prepare_packed_q8_0_tensor_layout<packed_dtype>(
          *matrix.tensor, matrix.rows, matrix.cols, packed.tensor, packed.storage)) {
    return false;
  }
  matrix.tensor = &packed.tensor;
  return true;
}

template <uint8_t packed_dtype>
inline bool prepare_packed_q8_0_output_logits_layout(native_backend & backend) noexcept {
  const auto * tensor = backend.output_native.tensor;
  if (tensor == nullptr) {
    return false;
  }

  if (!prepare_packed_q8_0_tensor_layout<packed_dtype>(
          *tensor,
          backend.output_native.rows,
          backend.output_native.cols,
          backend.output_packed_tensor,
          backend.output_packed_storage)) {
    return false;
  }

  backend.output.tensor = &backend.output_packed_tensor;
  backend.output_argmax = backend.output_native;
  return true;
}

inline bool prepare_packed_q8_0_output_logits(native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  return prepare_packed_q8_0_output_logits_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
      backend);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return prepare_packed_q8_0_output_logits_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
      backend);
#else
  (void) backend;
  return true;
#endif
}

inline bool prepare_prepared_output_logits(native_backend & backend) noexcept {
  const auto * tensor = backend.output_native.tensor;
  if (tensor == nullptr) {
    return false;
  }

  const uint64_t rows = static_cast<uint64_t>(backend.output_native.rows);
  const uint64_t cols = static_cast<uint64_t>(backend.output_native.cols);
  const uint64_t group_count = emel::kernel::detail::quant::packed_q6_k_x8_group_count(rows);
  const size_t prepared_group_bytes =
      emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(cols);
  const size_t argmax_prepared_group_bytes =
      emel::kernel::detail::quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(cols);
  const uint64_t prepared_storage_bytes =
      static_cast<uint64_t>(prepared_group_bytes) * group_count;
  const uint64_t argmax_prepared_storage_bytes =
      static_cast<uint64_t>(argmax_prepared_group_bytes) * group_count;
  if (prepared_group_bytes == 0u || prepared_storage_bytes == 0u ||
      argmax_prepared_group_bytes == 0u || argmax_prepared_storage_bytes == 0u) {
    return false;
  }

  backend.output_prepared_storage.resize(static_cast<size_t>(prepared_storage_bytes));
  backend.output_argmax_prepared_storage.resize(static_cast<size_t>(argmax_prepared_storage_bytes));
  const auto * src =
      reinterpret_cast<const emel::kernel::detail::quant::block_q6_k *>(tensor->data);
  if (!emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_prepared(
          src, rows, cols, backend.output_prepared_storage.data()) ||
      !emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_argmax_prepared(
          src, rows, cols, backend.output_argmax_prepared_storage.data())) {
    return false;
  }

  backend.output_prepared_tensor = *tensor;
  backend.output_prepared_tensor.type = emel::kernel::detail::dtype_q6_k_x8_q8_prepared;
  backend.output_prepared_tensor.data = backend.output_prepared_storage.data();
  backend.output_prepared_tensor.data_size = prepared_storage_bytes;
  backend.output.tensor = &backend.output_prepared_tensor;

  backend.output_argmax_prepared_tensor = *tensor;
  backend.output_argmax_prepared_tensor.type =
      emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
  backend.output_argmax_prepared_tensor.data = backend.output_argmax_prepared_storage.data();
  backend.output_argmax_prepared_tensor.data_size = argmax_prepared_storage_bytes;
  backend.output_argmax.tensor = &backend.output_argmax_prepared_tensor;
  return true;
}

inline bool prepare_packed_output_logits(native_backend & backend) noexcept {
  const auto * tensor = backend.output_native.tensor;
  if (tensor == nullptr) {
    return false;
  }

  const uint64_t rows = static_cast<uint64_t>(backend.output_native.rows);
  const uint64_t cols = static_cast<uint64_t>(backend.output_native.cols);
  const uint64_t group_count = emel::kernel::detail::quant::packed_q6_k_x8_group_count(rows);
  const size_t packed_group_bytes =
      emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(cols);
  const uint64_t packed_storage_bytes =
      static_cast<uint64_t>(packed_group_bytes) * group_count;
  if (packed_group_bytes == 0u || packed_storage_bytes == 0u) {
    return false;
  }

  backend.output_packed_storage.resize(static_cast<size_t>(packed_storage_bytes));
  if (!emel::kernel::detail::quant::pack_q6_k_rows_x8(
          reinterpret_cast<const emel::kernel::detail::quant::block_q6_k *>(tensor->data),
          rows,
          cols,
          backend.output_packed_storage.data())) {
    return false;
  }

  backend.output_packed_tensor = *tensor;
  backend.output_packed_tensor.type = emel::kernel::detail::dtype_q6_k_x8;
  backend.output_packed_tensor.data = backend.output_packed_storage.data();
  backend.output_packed_tensor.data_size = packed_storage_bytes;
  backend.output.tensor = &backend.output_packed_tensor;
  backend.output_argmax_packed_tensor = backend.output_packed_tensor;
  backend.output_argmax = backend.output;
  backend.output_argmax.tensor = &backend.output_argmax_packed_tensor;
  return true;
}

inline bool prepare_output_logits(native_backend & backend) noexcept {
  reset_output_logits(backend);

  const auto * tensor = backend.output_native.tensor;
  if (tensor == nullptr) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(tensor->type);
  if (dtype == emel::kernel::detail::dtype_q8_0) {
    return prepare_packed_q8_0_output_logits(backend);
  }
  if (dtype != emel::kernel::detail::dtype_q6_k) {
    return true;
  }

  if (prepared_q6_k_x8_q8_logits_supported(backend)) {
    return prepare_prepared_output_logits(backend);
  }

  if (packed_q6_k_x8_logits_supported(backend)) {
    return prepare_packed_output_logits(backend);
  }

  return true;
}

inline bool prepare_block_packed_q8_0_matrices(native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  for (auto & block : backend.blocks) {
    if (!prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.attention_q, block.attention_q_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.attention_k, block.attention_k_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.attention_v, block.attention_v_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.attention_output, block.attention_output_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.feed_forward_gate, block.feed_forward_gate_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.feed_forward_down, block.feed_forward_down_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
            block.feed_forward_up, block.feed_forward_up_packed)) {
      return false;
    }
  }
  return true;
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  for (auto & block : backend.blocks) {
    if (!prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.attention_q, block.attention_q_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.attention_k, block.attention_k_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.attention_v, block.attention_v_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.attention_output, block.attention_output_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.feed_forward_gate, block.feed_forward_gate_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.feed_forward_down, block.feed_forward_down_packed) ||
        !prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
            block.feed_forward_up, block.feed_forward_up_packed)) {
      return false;
    }
  }
  return true;
#else
  (void) backend;
  return true;
#endif
}

inline bool prepare_logits_input_q8_workspace(native_backend & backend) noexcept {
  backend.logits_input_q8_storage.clear();
  if ((!q8_logits_input_path_supported(backend.output) &&
       !q8_logits_input_path_supported(backend.output_argmax)) ||
      backend.n_embd <= 0 ||
      (backend.n_embd % static_cast<int32_t>(quant::QK_K)) != 0) {
    return true;
  }

  const size_t block_count =
      static_cast<size_t>(backend.n_embd) / static_cast<size_t>(quant::QK_K);
  if (block_count == 0u || block_count > quant::MAX_Q8_K_BLOCKS) {
    return false;
  }

  backend.logits_input_q8_storage.resize(block_count);
  return true;
}

inline bool update_packed_q8_0_input_requirement(const tensor_matrix & matrix,
                                                 size_t & max_block_count) noexcept {
  if (matrix.tensor == nullptr ||
      !emel::kernel::detail::is_packed_q8_0_vector_dtype(
          static_cast<uint8_t>(matrix.tensor->type))) {
    return true;
  }

  if (matrix.cols <= 0 ||
      (matrix.cols % static_cast<int32_t>(quant::QK8_0)) != 0) {
    return false;
  }

  const size_t block_count =
      static_cast<size_t>(matrix.cols) / static_cast<size_t>(quant::QK8_0);
  if (block_count == 0u || block_count > quant::MAX_Q8_0_BLOCKS) {
    return false;
  }

  max_block_count = std::max(max_block_count, block_count);
  return true;
}

inline bool prepare_packed_q8_0_input_workspace(native_backend & backend) noexcept {
  backend.packed_q8_0_input_storage.clear();

  size_t max_block_count = 0u;
  if (!update_packed_q8_0_input_requirement(backend.output, max_block_count)) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    if (!update_packed_q8_0_input_requirement(block.attention_q, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.attention_k, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.attention_v, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.attention_output, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.feed_forward_gate, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.feed_forward_down, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.feed_forward_up, max_block_count)) {
      return false;
    }
  }

  if (max_block_count == 0u) {
    return true;
  }

  backend.packed_q8_0_input_storage.resize(max_block_count);
  return true;
}

inline bool prepare_packed_q8_0_chunk4_input_workspace(native_backend & backend) noexcept {
  backend.packed_q8_0_chunk4_rows.clear();
  backend.packed_q8_0_chunk4_input_storage.clear();

  size_t max_block_count = 0u;
  if (!update_packed_q8_0_input_requirement(backend.output, max_block_count)) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    if (!update_packed_q8_0_input_requirement(block.attention_q, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.attention_k, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.attention_v, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.attention_output, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.feed_forward_gate, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.feed_forward_down, max_block_count) ||
        !update_packed_q8_0_input_requirement(block.feed_forward_up, max_block_count)) {
      return false;
    }
  }

  if (max_block_count == 0u) {
    return true;
  }

  const uint64_t chunk_row_count = static_cast<uint64_t>(k_prefill_q8_chunk_rows);
  const uint64_t cols = static_cast<uint64_t>(max_block_count) * quant::QK8_0;
  const size_t packed_bytes =
      emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(cols);
  if (packed_bytes == 0u) {
    return false;
  }

  backend.packed_q8_0_chunk4_rows.resize(chunk_row_count * max_block_count);
  backend.packed_q8_0_chunk4_input_storage.resize(packed_bytes);
  return true;
}

inline emel::kernel::event::tensor_view make_src_view(const tensor_matrix & matrix) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  const size_t row_bytes = row_storage_bytes(*matrix.tensor, matrix.cols);
  const uint64_t rows = static_cast<uint64_t>(matrix.rows);
  const bool packed_grouped =
      dtype == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
      dtype == emel::kernel::detail::dtype_q8_0_x4_bl8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
  const uint64_t storage_rows = dtype == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
          dtype == emel::kernel::detail::dtype_q8_0_x4_bl8
      ? emel::kernel::detail::quant::packed_q8_0_x4_group_count(rows)
      : packed_grouped
      ? emel::kernel::detail::quant::packed_q6_k_x8_group_count(rows)
      : rows;

  tensor.data = matrix.tensor->data;
  tensor.type = static_cast<emel::kernel::event::dtype>(matrix.tensor->type);
  tensor.ne = {static_cast<uint64_t>(matrix.cols), rows, 1u, 1u};
  tensor.nb[0] = dtype == emel::kernel::detail::dtype_f32 ? sizeof(float) : 1u;
  tensor.nb[1] = row_bytes;
  tensor.nb[2] = row_bytes * storage_rows;
  tensor.nb[3] = tensor.nb[2];
  return tensor;
}

inline uint64_t matrix_buffer_bytes(const tensor_matrix & matrix) noexcept {
  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  const uint64_t storage_rows = dtype == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
          dtype == emel::kernel::detail::dtype_q8_0_x4_bl8
      ? emel::kernel::detail::quant::packed_q8_0_x4_group_count(static_cast<uint64_t>(matrix.rows))
      : (dtype == emel::kernel::detail::dtype_q6_k_x8 ||
         dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
         dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared)
      ? emel::kernel::detail::quant::packed_q6_k_x8_group_count(
            static_cast<uint64_t>(matrix.rows))
      : static_cast<uint64_t>(matrix.rows);
  return static_cast<uint64_t>(row_storage_bytes(*matrix.tensor, matrix.cols)) * storage_rows;
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
    if (!block.attention_q_norm.empty()) {
      append_leaf_lifecycle_tensor(
          backend,
          block.attention_q_norm.data(),
          static_cast<uint64_t>(block.attention_q_norm.size()) * sizeof(float));
    }
    if (!block.attention_k_norm.empty()) {
      append_leaf_lifecycle_tensor(
          backend,
          block.attention_k_norm.data(),
          static_cast<uint64_t>(block.attention_k_norm.size()) * sizeof(float));
    }
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
      static_cast<uint64_t>(backend.key_cache.size()) * sizeof(uint16_t),
      1,
      false);
  backend.value_cache_tensor_id = append_lifecycle_tensor(
      backend,
      backend.value_cache.data(),
      static_cast<uint64_t>(backend.value_cache.size()) * sizeof(uint16_t),
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

inline bool prepare_packed_q8_0_input(native_backend & backend,
                                      std::span<const float> input) noexcept;

inline bool matmul_vector_prepared_packed_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    const int32_t input_cols,
    std::span<float> output) noexcept;

inline bool prepare_packed_q8_0_chunk4_input(native_backend & backend,
                                             std::span<const float> input,
                                             const int32_t input_cols) noexcept;

inline bool matmul_chunk4_prepared_packed_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    const int32_t input_cols,
    std::span<float> output) noexcept;

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

  if (packed_q8_0_input_path_supported(backend, matrix)) {
    return prepare_packed_q8_0_input(backend, input) &&
        matmul_vector_prepared_packed_q8_0_input(backend, matrix, matrix.cols, output);
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_src_view(
          input.data(),
          emel::kernel::event::dtype::f32,
          static_cast<uint64_t>(1u),
          static_cast<uint64_t>(input.size())),
      .dst = make_dst_view(
          output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
      .nth = 1,
  };
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(ev);
  backend.kernel_dispatch_calls += 1;
  backend.native_q8_0_dispatch_calls += static_cast<uint64_t>(
      matrix.tensor != nullptr &&
      static_cast<uint8_t>(matrix.tensor->type) == emel::kernel::detail::dtype_q8_0);
  return ok;
}

inline bool matmul_vector_argmax(native_backend & backend,
                                 const tensor_matrix & matrix,
                                 std::span<const float> input,
                                 int32_t & selected_index,
                                 float & selected_score) noexcept {
  if (matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat_argmax ev{
      .src0 = make_src_view(matrix),
      .src1 = make_src_view(
          input.data(),
          emel::kernel::event::dtype::f32,
          static_cast<uint64_t>(1u),
          static_cast<uint64_t>(input.size())),
      .dst = make_dst_view(&selected_score, static_cast<uint64_t>(1u), static_cast<uint64_t>(1u)),
      .nth = 1,
      .index_out = &selected_index,
  };
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(ev);
  backend.kernel_dispatch_calls += 1;
  backend.native_q8_0_dispatch_calls += static_cast<uint64_t>(
      matrix.tensor != nullptr &&
      static_cast<uint8_t>(matrix.tensor->type) == emel::kernel::detail::dtype_q8_0);
  return ok;
}

inline bool quantize_vector_q8_k(
    std::span<const float> input,
    std::span<emel::kernel::detail::quant::block_q8_k> output) noexcept {
  if ((input.size() % quant::QK_K) != 0u ||
      output.size() != input.size() / quant::QK_K) {
    return false;
  }

  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      input.data(),
      1u,
      output.data(),
      static_cast<int64_t>(input.size()));
  return true;
}

inline bool quantize_vector_q8_0(
    std::span<const float> input,
    std::span<emel::kernel::detail::quant::block_q8_0> output) noexcept {
  if ((input.size() % quant::QK8_0) != 0u ||
      output.size() != input.size() / quant::QK8_0) {
    return false;
  }

  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      input.data(),
      1u,
      output.data(),
      static_cast<int64_t>(input.size()));
  return true;
}

inline bool matmul_vector_q8_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    std::span<const emel::kernel::detail::quant::block_q8_k> input,
    const int32_t input_cols,
    std::span<float> output) noexcept {
  if (matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      input_cols != matrix.cols ||
      input_cols <= 0 ||
      static_cast<size_t>(matrix.rows) != output.size() ||
      (input_cols % static_cast<int32_t>(quant::QK_K)) != 0 ||
      static_cast<size_t>(input_cols / static_cast<int32_t>(quant::QK_K)) != input.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_q8_k_vector_view(input.data(), static_cast<uint64_t>(input_cols)),
      .dst = make_dst_view(
          output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
      .nth = 1,
  };
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(ev);
  backend.kernel_dispatch_calls += 1;
  return ok;
}

inline bool matmul_vector_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    std::span<const emel::kernel::detail::quant::block_q8_0> input,
    const int32_t input_cols,
    std::span<float> output) noexcept {
  if (matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      input_cols != matrix.cols ||
      input_cols <= 0 ||
      static_cast<size_t>(matrix.rows) != output.size() ||
      (input_cols % static_cast<int32_t>(quant::QK8_0)) != 0 ||
      static_cast<size_t>(input_cols / static_cast<int32_t>(quant::QK8_0)) != input.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_q8_0_vector_view(input.data(), static_cast<uint64_t>(input_cols)),
      .dst = make_dst_view(
          output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
      .nth = 1,
  };
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(ev);
  backend.kernel_dispatch_calls += 1;
  backend.packed_q8_0_dispatch_calls += static_cast<uint64_t>(
      matrix.tensor != nullptr &&
      emel::kernel::detail::is_packed_q8_0_vector_dtype(
          static_cast<uint8_t>(matrix.tensor->type)) &&
      ok);
  return ok;
}

inline bool prepare_packed_q8_0_input(native_backend & backend,
                                      std::span<const float> input) noexcept {
  if (backend.packed_q8_0_input_storage.empty() ||
      (input.size() % quant::QK8_0) != 0u) {
    return false;
  }

  const size_t block_count = input.size() / quant::QK8_0;
  if (block_count > backend.packed_q8_0_input_storage.size()) {
    return false;
  }

  return quantize_vector_q8_0(
      input,
      std::span<emel::kernel::detail::quant::block_q8_0>(
          backend.packed_q8_0_input_storage.data(), block_count));
}

inline bool prepare_packed_q8_0_chunk4_input(native_backend & backend,
                                             std::span<const float> input,
                                             const int32_t input_cols) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(input_cols);
  if (backend.packed_q8_0_chunk4_rows.empty() ||
      backend.packed_q8_0_chunk4_input_storage.empty() ||
      input_cols <= 0 ||
      input.size() != expected_size ||
      (static_cast<size_t>(input_cols) % quant::QK8_0) != 0u) {
    return false;
  }

  const size_t block_count = static_cast<size_t>(input_cols) / quant::QK8_0;
  const size_t required_rows =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * block_count;
  if (required_rows > backend.packed_q8_0_chunk4_rows.size()) {
    return false;
  }

  for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
    if (!quantize_vector_q8_0(
            input.subspan(
                static_cast<size_t>(row) * static_cast<size_t>(input_cols),
                static_cast<size_t>(input_cols)),
            std::span<emel::kernel::detail::quant::block_q8_0>(
                backend.packed_q8_0_chunk4_rows.data() +
                    (static_cast<size_t>(row) * block_count),
                block_count))) {
      return false;
    }
  }

  return emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      backend.packed_q8_0_chunk4_rows.data(),
      static_cast<uint64_t>(k_prefill_q8_chunk_rows),
      static_cast<uint64_t>(input_cols),
      backend.packed_q8_0_chunk4_input_storage.data());
}

inline bool matmul_vector_prepared_packed_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    const int32_t input_cols,
    std::span<float> output) noexcept {
  if (!packed_q8_0_input_path_supported(backend, matrix) ||
      input_cols <= 0 ||
      (input_cols % static_cast<int32_t>(quant::QK8_0)) != 0) {
    return false;
  }

  const size_t block_count =
      static_cast<size_t>(input_cols) / static_cast<size_t>(quant::QK8_0);
  if (block_count > backend.packed_q8_0_input_storage.size()) {
    return false;
  }

  return matmul_vector_q8_0_input(
      backend,
      matrix,
      std::span<const emel::kernel::detail::quant::block_q8_0>(
          backend.packed_q8_0_input_storage.data(), block_count),
      input_cols,
      output);
}

inline bool matmul_chunk4_prepared_packed_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    const int32_t input_cols,
    std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(matrix.rows);
  if (!packed_q8_0_chunk4_input_path_supported(backend, matrix) ||
      input_cols <= 0 ||
      input_cols != matrix.cols ||
      output.size() != expected_size) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_packed_q8_0_rhs_chunk4_view(
          backend.packed_q8_0_chunk4_input_storage.data(),
          static_cast<uint64_t>(input_cols)),
      .dst = make_batch_major_dst_view(
          output.data(),
          static_cast<uint64_t>(k_prefill_q8_chunk_rows),
          static_cast<uint64_t>(matrix.rows)),
      .nth = 1,
  };
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(ev);
  backend.kernel_dispatch_calls += 1;
  backend.packed_q8_0_dispatch_calls += static_cast<uint64_t>(ok);
  return ok;
}

inline bool matmul_vector_q8_input_argmax(
    native_backend & backend,
    const tensor_matrix & matrix,
    std::span<const emel::kernel::detail::quant::block_q8_k> input,
    const int32_t input_cols,
    int32_t & selected_index,
    float & selected_score) noexcept {
  if (matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      input_cols != matrix.cols ||
      input_cols <= 0 ||
      (input_cols % static_cast<int32_t>(quant::QK_K)) != 0 ||
      static_cast<size_t>(input_cols / static_cast<int32_t>(quant::QK_K)) != input.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat_argmax ev{
      .src0 = make_src_view(matrix),
      .src1 = make_q8_k_vector_view(input.data(), static_cast<uint64_t>(input_cols)),
      .dst = make_dst_view(&selected_score, static_cast<uint64_t>(1u), static_cast<uint64_t>(1u)),
      .nth = 1,
      .index_out = &selected_index,
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
    square_sum += static_cast<double>(value * value);
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

template <class value_type>
inline std::span<value_type> chunk4_row_span(std::span<value_type> values,
                                             const int32_t row,
                                             const int32_t cols) noexcept {
  return values.subspan(
      static_cast<size_t>(row) * static_cast<size_t>(cols),
      static_cast<size_t>(cols));
}

template <class value_type>
inline std::span<const value_type> chunk4_row_span(std::span<const value_type> values,
                                                   const int32_t row,
                                                   const int32_t cols) noexcept {
  return values.subspan(
      static_cast<size_t>(row) * static_cast<size_t>(cols),
      static_cast<size_t>(cols));
}

inline bool rms_norm_chunk4(std::span<const float> input,
                            const int32_t cols,
                            std::span<const float> weight,
                            const float epsilon,
                            std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(cols);
  if (cols <= 0 || input.size() != expected_size || output.size() != expected_size) {
    return false;
  }

  for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
    if (!rms_norm(
            chunk4_row_span<const float>(input, row, cols),
            weight,
            epsilon,
            chunk4_row_span<float>(output, row, cols))) {
      return false;
    }
  }
  return true;
}

inline bool add_chunk4_rows_in_place(std::span<float> dst,
                                     std::span<const float> src,
                                     const int32_t cols) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(cols);
  if (cols <= 0 || dst.size() != expected_size || src.size() != expected_size) {
    return false;
  }

  for (size_t idx = 0; idx < expected_size; ++idx) {
    dst[idx] += src[idx];
  }
  return true;
}

inline float silu(const float value) noexcept;

inline bool apply_silu_mul_chunk4(std::span<const float> gate,
                                  std::span<const float> up,
                                  const int32_t cols,
                                  std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(cols);
  if (cols <= 0 || gate.size() != expected_size || up.size() != expected_size ||
      output.size() != expected_size) {
    return false;
  }

  for (size_t idx = 0; idx < expected_size; ++idx) {
    output[idx] = silu(gate[idx]) * up[idx];
  }
  return true;
}

inline void round_q_for_nonflash(std::span<const float> q_source,
                                 std::span<float> q_target) noexcept {
  for (size_t idx = 0; idx < q_source.size(); ++idx) {
    q_target[idx] = quant::fp16_to_fp32(quant::fp32_to_fp16(q_source[idx]));
  }
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

  const float theta_scale = ::powf(rope_freq_base, -2.0f / static_cast<float>(rot_dim));
  for (int32_t head = 0; head < head_count; ++head) {
    float * head_ptr =
        vector.data() + (static_cast<size_t>(head) * static_cast<size_t>(head_dim));
    float theta = static_cast<float>(position);
    for (int32_t dim = 0; dim + 1 < rot_dim; dim += 2) {
      const float cos_theta = ::cosf(theta);
      const float sin_theta = ::sinf(theta);
      const float x0 = head_ptr[dim];
      const float x1 = head_ptr[dim + 1];
      head_ptr[dim] = x0 * cos_theta - x1 * sin_theta;
      head_ptr[dim + 1] = x0 * sin_theta + x1 * cos_theta;
      theta *= theta_scale;
    }
  }
}

#if defined(__ARM_NEON) && defined(__aarch64__)
inline float32x4_t silu4_neon(float32x4_t x) noexcept {
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t neg_x = vsubq_f32(zero, x);
  const float32x4_t exp_neg_x = ::emel::kernel::detail::expf4_ggml(neg_x);
  const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
  return vdivq_f32(x, one_plus_exp_neg_x);
}
#endif

inline float silu(const float value) noexcept {
#if defined(__ARM_NEON) && defined(__aarch64__)
  return vgetq_lane_f32(silu4_neon(vdupq_n_f32(value)), 0);
#else
  return value / (1.0f + ::expf(-value));
#endif
}

inline size_t layer_cache_offset(const native_backend & backend,
                                 const int32_t layer,
                                 const int32_t position,
                                 const int32_t kv_dim) noexcept {
  return ((static_cast<size_t>(layer) * static_cast<size_t>(backend.n_ctx)) +
          static_cast<size_t>(position)) *
         static_cast<size_t>(kv_dim);
}

inline size_t flash_layer_cache_layer_offset(const native_backend & backend,
                                             const int32_t layer,
                                             const int32_t kv_head_dim) noexcept {
  return static_cast<size_t>(layer) *
      static_cast<size_t>(backend.n_head_kv) *
      static_cast<size_t>(backend.n_ctx) *
      static_cast<size_t>(kv_head_dim);
}

inline size_t flash_layer_cache_head_offset(const native_backend & backend,
                                            const int32_t layer,
                                            const int32_t kv_head,
                                            const int32_t kv_head_dim) noexcept {
  return flash_layer_cache_layer_offset(backend, layer, kv_head_dim) +
      static_cast<size_t>(kv_head) *
      static_cast<size_t>(backend.n_ctx) *
      static_cast<size_t>(kv_head_dim);
}

inline size_t flash_layer_cache_head_position_offset(const native_backend & backend,
                                                     const int32_t layer,
                                                     const int32_t kv_head,
                                                     const int32_t position,
                                                     const int32_t kv_head_dim) noexcept {
  return flash_layer_cache_head_offset(backend, layer, kv_head, kv_head_dim) +
      static_cast<size_t>(position) * static_cast<size_t>(kv_head_dim);
}

inline void store_fp16_rounded_cache(std::span<const float> src, uint16_t * dst) noexcept {
  for (size_t idx = 0; idx < src.size(); ++idx) {
    dst[idx] = quant::fp32_to_fp16(src[idx]);
  }
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

inline void fill_masked_softmax_probs_ggml(std::span<const float> scores,
                                           const int32_t active_count,
                                           std::span<float> probs_out,
                                           std::span<float> rounded_probs_out) noexcept {
  const int32_t width = static_cast<int32_t>(scores.size());
  if (width <= 0 || active_count <= 0 || probs_out.size() != scores.size() ||
      rounded_probs_out.size() != scores.size() || active_count > width) {
    std::fill(probs_out.begin(), probs_out.end(), 0.0f);
    std::fill(rounded_probs_out.begin(), rounded_probs_out.end(), 0.0f);
    return;
  }

  float max_score = -std::numeric_limits<float>::infinity();
  for (int32_t position = 0; position < width; ++position) {
    max_score = std::max(max_score, scores[static_cast<size_t>(position)]);
  }

  const double score_sum = ::emel::kernel::detail::exp_and_sum_ggml_f32(
      scores.data(), probs_out.data(), static_cast<uint64_t>(width), max_score);

  const float inv_score_sum =
      score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
  for (int32_t position = 0; position < width; ++position) {
    const float weight = probs_out[static_cast<size_t>(position)] * inv_score_sum;
    probs_out[static_cast<size_t>(position)] = weight;
    rounded_probs_out[static_cast<size_t>(position)] =
        emel::kernel::detail::round_fp16_weight(weight);
  }
}

inline bool compute_attention(native_backend & backend,
                              const int32_t layer_index,
                              const int32_t position_limit,
                              const std::span<const float> q_vector) noexcept {
  const int32_t head_count = backend.n_head;
  const int32_t kv_head_count = backend.n_head_kv;
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const int32_t q_dim = head_count * head_dim;
  const int32_t kv_dim = kv_head_count * kv_head_dim;
  const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int32_t attn_width = backend.n_ctx;

  if (position_limit <= 0 || position_limit > attn_width ||
      q_vector.size() != static_cast<size_t>(q_dim) ||
      backend.attn_ctx.size() != static_cast<size_t>(q_dim) ||
      backend.attn_scores.size() != static_cast<size_t>(attn_width) ||
      backend.attn_probs.size() != static_cast<size_t>(attn_width) ||
      backend.attn_probs_rounded.size() != static_cast<size_t>(attn_width) ||
      backend.attn_value_column.size() != static_cast<size_t>(attn_width)) {
    return false;
  }

  std::fill(backend.attn_ctx.begin(), backend.attn_ctx.end(), 0.0f);
  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);
    const size_t kv_offset = static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_dim);

    for (int32_t position = 0; position < position_limit; ++position) {
      const size_t cache_offset =
          layer_cache_offset(backend, layer_index, position, kv_dim) + kv_offset;
      const float score = emel::kernel::detail::dot_product_f32_f16_scores(
          q_vector.data() + static_cast<std::ptrdiff_t>(q_offset),
          backend.key_cache.data() + static_cast<std::ptrdiff_t>(cache_offset),
          static_cast<uint64_t>(head_dim)) *
          inv_scale;
      backend.attn_scores[static_cast<size_t>(position)] = score;
    }

    for (int32_t position = position_limit; position < attn_width; ++position) {
      backend.attn_scores[static_cast<size_t>(position)] = -std::numeric_limits<float>::infinity();
    }

    fill_masked_softmax_probs_ggml(backend.attn_scores,
                                   position_limit,
                                   backend.attn_probs,
                                   backend.attn_probs_rounded);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      const size_t cache_offset = q_offset + static_cast<size_t>(dim);
      for (int32_t position = 0; position < position_limit; ++position) {
        const size_t value_offset =
            layer_cache_offset(backend, layer_index, position, kv_dim) + kv_offset;
        backend.attn_value_column[static_cast<size_t>(position)] =
            quant::fp16_to_fp32(
                backend.value_cache[value_offset + static_cast<size_t>(dim)]);
      }
      for (int32_t position = position_limit; position < attn_width; ++position) {
        backend.attn_value_column[static_cast<size_t>(position)] = 0.0f;
      }
      backend.attn_ctx[cache_offset] = emel::kernel::detail::dot_product_ggml_f16_scores(
          backend.attn_value_column.data(),
          backend.attn_probs_rounded.data(),
          static_cast<uint64_t>(attn_width));
    }
  }

  return true;
}

inline emel::kernel::event::op_flash_attn_ext make_flash_attn_request(
    const native_backend & backend,
    const int32_t layer_index,
    const int32_t position) noexcept {
  emel::kernel::event::op_flash_attn_ext request{};
  const uint64_t kv_tokens = static_cast<uint64_t>(position + 1);
  const uint64_t head_dim = static_cast<uint64_t>(backend.head_dim);
  const uint64_t head_count = static_cast<uint64_t>(backend.n_head);
  const uint64_t kv_head_dim = static_cast<uint64_t>(backend.head_dim_kv);
  const uint64_t kv_head_count = static_cast<uint64_t>(backend.n_head_kv);
  const size_t layer_offset = flash_layer_cache_layer_offset(
      backend, layer_index, backend.head_dim_kv);
  const float scale = 1.0f / std::sqrt(static_cast<float>(backend.head_dim));
  const uint32_t masked_total_tokens = static_cast<uint32_t>(backend.n_ctx);

  request.src0 = make_src_view_3d(
      const_cast<float *>(backend.q.data()),
      emel::kernel::event::dtype::f32,
      head_dim,
      1u,
      head_count);
  request.src1 = make_src_view_strided_3d(
      const_cast<uint16_t *>(backend.flash_key_cache.data() + layer_offset),
      emel::kernel::event::dtype::f16,
      kv_head_dim,
      kv_tokens,
      kv_head_count,
      sizeof(uint16_t) * kv_head_dim,
      sizeof(uint16_t) * static_cast<uint64_t>(backend.n_ctx) * kv_head_dim);
  request.src2 = make_src_view_strided_3d(
      const_cast<uint16_t *>(backend.flash_value_cache.data() + layer_offset),
      emel::kernel::event::dtype::f16,
      kv_head_dim,
      kv_tokens,
      kv_head_count,
      sizeof(uint16_t) * kv_head_dim,
      sizeof(uint16_t) * static_cast<uint64_t>(backend.n_ctx) * kv_head_dim);
  request.dst = make_dst_view_3d(
      const_cast<float *>(backend.attn_ctx.data()), head_dim, 1u, head_count);
  request.nth = 1;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale),
              &masked_total_tokens,
              sizeof(masked_total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(masked_total_tokens);
  return request;
}

inline bool flash_attention_supported(const native_backend & backend,
                                      const int32_t position) noexcept {
  if (backend.n_layer <= 0 || position < 0 || position >= backend.n_ctx) {
    return false;
  }
  return emel::kernel::detail::can_run_flash_attn_ext(
      make_flash_attn_request(backend, 0, position));
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

inline bool store_attention_kv_cache(native_backend & backend,
                                     const int32_t layer_index,
                                     const int32_t position,
                                     std::span<const float> k_vector,
                                     std::span<const float> v_vector) noexcept {
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  if (position < 0 ||
      position >= backend.n_ctx ||
      k_vector.size() != static_cast<size_t>(kv_dim) ||
      v_vector.size() != static_cast<size_t>(kv_dim)) {
    return false;
  }

  const size_t cache_offset = layer_cache_offset(backend, layer_index, position, kv_dim);
  store_fp16_rounded_cache(k_vector, backend.key_cache.data() + cache_offset);
  store_fp16_rounded_cache(v_vector, backend.value_cache.data() + cache_offset);

  for (int32_t kv_head = 0; kv_head < backend.n_head_kv; ++kv_head) {
    const size_t src_offset =
        static_cast<size_t>(kv_head) * static_cast<size_t>(backend.head_dim_kv);
    const size_t flash_cache_offset = flash_layer_cache_head_position_offset(
        backend, layer_index, kv_head, position, backend.head_dim_kv);
    store_fp16_rounded_cache(
        k_vector.subspan(src_offset, static_cast<size_t>(backend.head_dim_kv)),
        backend.flash_key_cache.data() + flash_cache_offset);
    store_fp16_rounded_cache(
        v_vector.subspan(src_offset, static_cast<size_t>(backend.head_dim_kv)),
        backend.flash_value_cache.data() + flash_cache_offset);
  }

  return true;
}

template <emel::generator::attention_mode mode>
inline bool run_attention_for_q_vector(native_backend & backend,
                                       const int32_t layer_index,
                                       const int32_t position,
                                       std::span<const float> q_vector) noexcept {
  if constexpr (mode == emel::generator::attention_mode::flash) {
    if (q_vector.size() != backend.q.size()) {
      return false;
    }
    std::copy(q_vector.begin(), q_vector.end(), backend.q.begin());
    return dispatch_flash_attention(backend, layer_index, position);
  } else {
    if (q_vector.size() != backend.q_attn.size()) {
      return false;
    }
    round_q_for_nonflash(q_vector, backend.q_attn);
    return compute_attention(backend, layer_index, position + 1, backend.q_attn);
  }
}

template <emel::generator::attention_mode mode>
inline bool run_attention(native_backend & backend,
                          const int32_t layer_index,
                          const int32_t position) noexcept {
  return run_attention_for_q_vector<mode>(backend, layer_index, position, backend.q);
}

template <emel::generator::attention_mode mode>
inline bool run_layer(native_backend & backend,
                      const int32_t layer_index,
                      const int32_t position) noexcept {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!rms_norm(backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  const bool qkv_shared_packed_q8_0 =
      packed_q8_0_input_path_supported(backend, block.attention_q) &&
      packed_q8_0_input_path_supported(backend, block.attention_k) &&
      packed_q8_0_input_path_supported(backend, block.attention_v);
  if (qkv_shared_packed_q8_0) {
    if (!prepare_packed_q8_0_input(backend, backend.norm) ||
        !matmul_vector_prepared_packed_q8_0_input(
            backend, block.attention_q, block.attention_q.cols, backend.q) ||
        !matmul_vector_prepared_packed_q8_0_input(
            backend, block.attention_k, block.attention_k.cols, backend.k) ||
        !matmul_vector_prepared_packed_q8_0_input(
            backend, block.attention_v, block.attention_v.cols, backend.v)) {
      return false;
    }
  } else if (!matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
             !matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
             !matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
    return false;
  }

  if (is_qwen3_runtime(backend) && !apply_qwen3_attention_qk_norm(backend, block)) {
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
  if (!store_attention_kv_cache(
          backend,
          layer_index,
          position,
          std::span<const float>(backend.k.data(), backend.k.size()),
          std::span<const float>(backend.v.data(), backend.v.size())) ||
      !run_attention<mode>(backend, layer_index, position) ||
      !matmul_vector(backend, block.attention_output, backend.attn_ctx, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  if (!rms_norm(backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  const bool gate_up_shared_packed_q8_0 =
      packed_q8_0_input_path_supported(backend, block.feed_forward_gate) &&
      packed_q8_0_input_path_supported(backend, block.feed_forward_up);
  if (gate_up_shared_packed_q8_0) {
    if (!prepare_packed_q8_0_input(backend, backend.norm) ||
        !matmul_vector_prepared_packed_q8_0_input(
            backend, block.feed_forward_gate, block.feed_forward_gate.cols, backend.gate) ||
        !matmul_vector_prepared_packed_q8_0_input(
            backend, block.feed_forward_up, block.feed_forward_up.cols, backend.up)) {
      return false;
    }
  } else if (!matmul_vector(backend, block.feed_forward_gate, backend.norm, backend.gate) ||
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

inline bool run_layer_flash(native_backend & backend,
                            const int32_t layer_index,
                            const int32_t position) noexcept {
  return run_layer<emel::generator::attention_mode::flash>(backend, layer_index, position);
}

inline bool run_layer_nonflash(native_backend & backend,
                               const int32_t layer_index,
                               const int32_t position) noexcept {
  return run_layer<emel::generator::attention_mode::nonflash>(backend, layer_index, position);
}

inline bool compute_logits(native_backend & backend) noexcept {
  if (!rms_norm(backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  const bool packed_q8_0_logits_path =
      packed_q8_0_input_path_supported(backend, backend.output);
  if (packed_q8_0_logits_path) {
    return prepare_packed_q8_0_input(backend, backend.norm) &&
        matmul_vector_prepared_packed_q8_0_input(
               backend, backend.output, backend.n_embd, backend.bound_logits);
  }

  const bool packed_q8_logits_path =
      !backend.logits_input_q8_storage.empty() && q8_logits_input_path_supported(backend.output);
  if (packed_q8_logits_path) {
    return quantize_vector_q8_k(backend.norm, backend.logits_input_q8_storage) &&
        matmul_vector_q8_input(
               backend,
               backend.output,
               backend.logits_input_q8_storage,
               backend.n_embd,
               backend.bound_logits);
  }

  return matmul_vector(backend, backend.output, backend.norm, backend.bound_logits);
}

inline bool compute_logits_preselected_argmax(native_backend & backend,
                                              int32_t & selected_index,
                                              float & selected_score) noexcept {
  if (!rms_norm(backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  const tensor_matrix & output_matrix =
      backend.output_argmax.tensor != nullptr ? backend.output_argmax : backend.output;
  const bool packed_q8_logits_path =
      !backend.logits_input_q8_storage.empty() && q8_logits_input_path_supported(output_matrix);
  if (packed_q8_logits_path) {
    return quantize_vector_q8_k(backend.norm, backend.logits_input_q8_storage) &&
        matmul_vector_q8_input_argmax(
               backend,
               output_matrix,
               backend.logits_input_q8_storage,
               backend.n_embd,
               selected_index,
               selected_score);
  }

  return matmul_vector_argmax(backend, output_matrix, backend.norm, selected_index, selected_score);
}

template <emel::generator::attention_mode mode>
inline bool run_prefill_scalar_tokens(native_backend & backend,
                                      const size_t token_begin,
                                      const size_t token_end) noexcept {
  for (size_t token_index = token_begin; token_index < token_end; ++token_index) {
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
      if (!run_layer<mode>(backend, layer, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return true;
}

template <emel::generator::attention_mode mode>
inline bool run_layer_chunk4(native_backend & backend,
                             const int32_t layer_index,
                             const size_t token_base) noexcept {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  const int32_t q_dim = backend.n_head * backend.head_dim;
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;
  const int32_t ffn_dim = block.feed_forward_gate.rows;

  if (!rms_norm_chunk4(
          backend.hidden_chunk4,
          backend.n_embd,
          block.attention_norm,
          backend.rms_epsilon,
          backend.norm_chunk4) ||
      !prepare_packed_q8_0_chunk4_input(backend, backend.norm_chunk4, backend.n_embd) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend, block.attention_q, block.attention_q.cols, backend.q_chunk4) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend, block.attention_k, block.attention_k.cols, backend.k_chunk4) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend, block.attention_v, block.attention_v.cols, backend.v_chunk4)) {
    return false;
  }

  const bool qwen3_runtime = is_qwen3_runtime(backend);
  for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
    const int32_t position = backend.bound_positions[token_base + static_cast<size_t>(row)];
    auto q_row = chunk4_row_span<float>(std::span<float>(backend.q_chunk4), row, q_dim);
    auto k_row = chunk4_row_span<float>(std::span<float>(backend.k_chunk4), row, kv_dim);
    const auto v_row = chunk4_row_span<const float>(
        std::span<const float>(backend.v_chunk4), row, kv_dim);

    if (qwen3_runtime &&
        (!apply_headwise_rms_norm(
             q_row,
             block.attention_q_norm,
             backend.n_head,
             backend.head_dim,
             backend.rms_epsilon) ||
         !apply_headwise_rms_norm(
             k_row,
             block.attention_k_norm,
             backend.n_head_kv,
             backend.head_dim_kv,
             backend.rms_epsilon))) {
      return false;
    }

    apply_rope(
        q_row, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
    apply_rope(k_row,
               backend.n_head_kv,
               backend.head_dim_kv,
               backend.n_rot,
               position,
               backend.rope_freq_base);

    if (!store_attention_kv_cache(backend, layer_index, position, k_row, v_row) ||
        !run_attention_for_q_vector<mode>(backend, layer_index, position, q_row)) {
      return false;
    }

    std::copy(
        backend.attn_ctx.begin(),
        backend.attn_ctx.end(),
        chunk4_row_span<float>(std::span<float>(backend.attn_ctx_chunk4), row, q_dim).begin());
    backend.kv_cache_tokens = position + 1;
  }

  if (!prepare_packed_q8_0_chunk4_input(backend, backend.attn_ctx_chunk4, q_dim) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend,
          block.attention_output,
          block.attention_output.cols,
          backend.projected_chunk4) ||
      !add_chunk4_rows_in_place(backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd) ||
      !rms_norm_chunk4(
          backend.hidden_chunk4,
          backend.n_embd,
          block.feed_forward_norm,
          backend.rms_epsilon,
          backend.norm_chunk4) ||
      !prepare_packed_q8_0_chunk4_input(backend, backend.norm_chunk4, backend.n_embd) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend,
          block.feed_forward_gate,
          block.feed_forward_gate.cols,
          backend.gate_chunk4) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend,
          block.feed_forward_up,
          block.feed_forward_up.cols,
          backend.up_chunk4) ||
      !apply_silu_mul_chunk4(
          backend.gate_chunk4, backend.up_chunk4, ffn_dim, backend.ffn_hidden_chunk4) ||
      !prepare_packed_q8_0_chunk4_input(backend, backend.ffn_hidden_chunk4, ffn_dim) ||
      !matmul_chunk4_prepared_packed_q8_0_input(
          backend,
          block.feed_forward_down,
          block.feed_forward_down.cols,
          backend.projected_chunk4) ||
      !add_chunk4_rows_in_place(backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd)) {
    return false;
  }

  return true;
}

template <emel::generator::attention_mode mode>
inline bool run_prefill_chunk4_tokens(native_backend & backend,
                                      const size_t token_limit) noexcept {
  for (size_t token_base = 0; token_base < token_limit;
       token_base += static_cast<size_t>(k_prefill_q8_chunk_rows)) {
    for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
      const size_t token_index = token_base + static_cast<size_t>(row);
      const int32_t token_id = backend.bound_tokens[token_index];
      const int32_t position = backend.bound_positions[token_index];
      if (token_id < 0 ||
          token_id >= backend.token_embedding.rows ||
          position < 0 ||
          position >= backend.n_ctx ||
          !copy_tensor_row(
              *backend.token_embedding.tensor,
              token_id,
              chunk4_row_span<float>(
                  std::span<float>(backend.hidden_chunk4), row, backend.n_embd))) {
        return false;
      }
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_chunk4<mode>(backend, layer, token_base)) {
        return false;
      }
    }

    std::copy(
        chunk4_row_span<const float>(
            std::span<const float>(backend.hidden_chunk4),
            k_prefill_q8_chunk_rows - 1,
            backend.n_embd)
            .begin(),
        chunk4_row_span<const float>(
            std::span<const float>(backend.hidden_chunk4),
            k_prefill_q8_chunk_rows - 1,
            backend.n_embd)
            .end(),
        backend.hidden.begin());
  }

  return true;
}

template <emel::generator::attention_mode mode>
inline bool run_prefill(native_backend & backend) noexcept {
  backend.kv_cache_tokens = 0;

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  if (!run_prefill_scalar_tokens<mode>(backend, 0u, token_count)) {
    return false;
  }

  return compute_logits(backend);
}

template <emel::generator::attention_mode mode>
inline bool run_prefill_chunk4(native_backend & backend) noexcept {
  backend.kv_cache_tokens = 0;

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_limit =
      token_count - (token_count % static_cast<size_t>(k_prefill_q8_chunk_rows));
  if (chunk_limit == 0u ||
      !run_prefill_chunk4_tokens<mode>(backend, chunk_limit) ||
      !run_prefill_scalar_tokens<mode>(backend, chunk_limit, token_count)) {
    return false;
  }

  return compute_logits(backend);
}

inline bool run_prefill_flash(native_backend & backend) noexcept {
  return run_prefill<emel::generator::attention_mode::flash>(backend);
}

inline bool run_prefill_nonflash(native_backend & backend) noexcept {
  return run_prefill<emel::generator::attention_mode::nonflash>(backend);
}

template <emel::generator::attention_mode mode>
inline bool run_prefill_preselected_argmax(native_backend & backend,
                                           int32_t & selected_index,
                                           float & selected_score) noexcept {
  backend.kv_cache_tokens = 0;
  if (!run_prefill_scalar_tokens<mode>(
          backend, 0u, static_cast<size_t>(backend.bound_token_count))) {
    return false;
  }

  return compute_logits_preselected_argmax(backend, selected_index, selected_score);
}

template <emel::generator::attention_mode mode>
inline bool run_prefill_chunk4_preselected_argmax(native_backend & backend,
                                                  int32_t & selected_index,
                                                  float & selected_score) noexcept {
  backend.kv_cache_tokens = 0;

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_limit =
      token_count - (token_count % static_cast<size_t>(k_prefill_q8_chunk_rows));
  if (chunk_limit == 0u ||
      !run_prefill_chunk4_tokens<mode>(backend, chunk_limit) ||
      !run_prefill_scalar_tokens<mode>(backend, chunk_limit, token_count)) {
    return false;
  }

  return compute_logits_preselected_argmax(backend, selected_index, selected_score);
}

template <emel::generator::attention_mode mode>
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
    if (!run_layer<mode>(backend, layer, position)) {
      return false;
    }
  }
  backend.kv_cache_tokens = position + 1;
  return compute_logits(backend);
}

template <emel::generator::attention_mode mode>
inline bool run_decode_preselected_argmax(native_backend & backend,
                                          const emel::graph::processor::event::execute & request,
                                          int32_t & selected_index,
                                          float & selected_score) noexcept {
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
    if (!run_layer<mode>(backend, layer, position)) {
      return false;
    }
  }
  backend.kv_cache_tokens = position + 1;
  return compute_logits_preselected_argmax(backend, selected_index, selected_score);
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

  backend.quantized_audit =
      emel::model::llama::detail::build_quantized_path_audit(backend.execution);
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
      !bind_output_projection(backend) ||
      !prepare_output_logits(backend)) {
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
  const bool qwen3_runtime = is_qwen3_runtime(backend);
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
        (qwen3_runtime &&
         !dequantize_tensor_vector(*block.attention_q_norm.tensor, weights.attention_q_norm)) ||
        (qwen3_runtime &&
         !dequantize_tensor_vector(*block.attention_k_norm.tensor, weights.attention_k_norm)) ||
        !bind_tensor_rows(*block.attention_output.tensor, weights.attention_output) ||
        !dequantize_tensor_vector(*block.feed_forward_norm.tensor, weights.feed_forward_norm) ||
        !bind_tensor_rows(*block.feed_forward_gate.tensor, weights.feed_forward_gate) ||
        !bind_tensor_rows(*block.feed_forward_down.tensor, weights.feed_forward_down) ||
        !bind_tensor_rows(*block.feed_forward_up.tensor, weights.feed_forward_up)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  if (!prepare_block_packed_q8_0_matrices(backend) ||
      !prepare_logits_input_q8_workspace(backend) ||
      !prepare_packed_q8_0_input_workspace(backend) ||
      !prepare_packed_q8_0_chunk4_input_workspace(backend)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const int32_t declared_key_length = model_data.params.attention_key_length;
  const int32_t declared_value_length = model_data.params.attention_value_length;
  backend.head_dim = declared_key_length > 0 ? declared_key_length : backend.n_embd / backend.n_head;
  backend.head_dim_kv =
      declared_key_length > 0 ? declared_key_length : backend.blocks[0].attention_k.rows / backend.n_head_kv;
  backend.n_rep = backend.n_head / backend.n_head_kv;
  const int32_t q_dim = backend.n_head * backend.head_dim;
  const int32_t kv_dim = backend.n_head_kv * backend.head_dim_kv;

  if (backend.head_dim_kv <= 0 ||
      backend.head_dim <= 0 ||
      backend.n_rep <= 0 ||
      (declared_value_length > 0 && declared_value_length != backend.head_dim_kv) ||
      backend.blocks[0].attention_q.cols != backend.n_embd ||
      backend.blocks[0].attention_q.rows != q_dim ||
      backend.blocks[0].attention_k.cols != backend.n_embd ||
      backend.blocks[0].attention_k.rows != kv_dim ||
      backend.blocks[0].attention_v.cols != backend.n_embd ||
      backend.blocks[0].attention_v.rows != kv_dim ||
      backend.blocks[0].attention_output.cols != q_dim ||
      backend.blocks[0].attention_output.rows != backend.n_embd ||
      static_cast<int32_t>(backend.blocks[0].attention_norm.size()) != backend.n_embd ||
      static_cast<int32_t>(backend.blocks[0].feed_forward_norm.size()) != backend.n_embd ||
      backend.blocks[0].feed_forward_gate.cols != backend.n_embd ||
      backend.blocks[0].feed_forward_up.cols != backend.n_embd ||
      backend.blocks[0].feed_forward_down.rows != backend.n_embd) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  if (qwen3_runtime) {
    for (const auto & block : backend.blocks) {
      if (static_cast<int32_t>(block.attention_q_norm.size()) != backend.head_dim ||
          static_cast<int32_t>(block.attention_k_norm.size()) != backend.head_dim_kv) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
    }
  }

  backend.key_cache.resize(static_cast<size_t>(backend.n_layer) *
                           static_cast<size_t>(backend.n_ctx) *
                           static_cast<size_t>(kv_dim));
  backend.value_cache.resize(static_cast<size_t>(backend.n_layer) *
                             static_cast<size_t>(backend.n_ctx) *
                             static_cast<size_t>(kv_dim));
  backend.flash_key_cache.resize(static_cast<size_t>(backend.n_layer) *
                                 static_cast<size_t>(backend.n_ctx) *
                                 static_cast<size_t>(kv_dim));
  backend.flash_value_cache.resize(static_cast<size_t>(backend.n_layer) *
                                   static_cast<size_t>(backend.n_ctx) *
                                   static_cast<size_t>(kv_dim));
  backend.bound_logits.resize(static_cast<size_t>(backend.n_vocab));
  backend.bound_tokens.resize(static_cast<size_t>(backend.n_ctx));
  backend.bound_positions.resize(static_cast<size_t>(backend.n_ctx));
  backend.hidden.resize(static_cast<size_t>(backend.n_embd));
  backend.hidden_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(backend.n_embd));
  backend.norm.resize(static_cast<size_t>(backend.n_embd));
  backend.norm_chunk4.resize(backend.hidden_chunk4.size());
  backend.q.resize(static_cast<size_t>(q_dim));
  backend.q_attn.resize(static_cast<size_t>(q_dim));
  backend.q_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(q_dim));
  backend.k.resize(static_cast<size_t>(kv_dim));
  backend.k_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(kv_dim));
  backend.v.resize(static_cast<size_t>(kv_dim));
  backend.v_chunk4.resize(backend.k_chunk4.size());
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs_rounded.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_value_column.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(static_cast<size_t>(q_dim));
  backend.attn_ctx_chunk4.resize(backend.q_chunk4.size());
  backend.projected.resize(static_cast<size_t>(backend.n_embd));
  backend.projected_chunk4.resize(backend.hidden_chunk4.size());
  backend.gate.resize(static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));
  backend.gate_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) *
      static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));
  backend.up.resize(static_cast<size_t>(backend.blocks[0].feed_forward_up.rows));
  backend.up_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) *
      static_cast<size_t>(backend.blocks[0].feed_forward_up.rows));
  backend.ffn_hidden.resize(static_cast<size_t>(backend.blocks[0].feed_forward_gate.rows));
  backend.ffn_hidden_chunk4.resize(backend.gate_chunk4.size());
  build_lifecycle(backend);

  return emel::error::cast(emel::model::loader::error::none);
}

inline uint32_t quantized_contract_stage_count(
    const native_backend & backend,
    const emel::model::llama::detail::quantized_contract_kind kind) noexcept {
  uint32_t count = 0u;
  for (const auto & stage : backend.quantized_audit.stages) {
    count += static_cast<uint32_t>(stage.contract == kind);
  }
  return count;
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

inline bool validate_preselected_argmax(const emel::graph::processor::event::execute & request,
                                        int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  const auto * plan = request_plan(request, err_out);
  if (plan == nullptr || !check_backend(backend, err_out)) {
    return false;
  }

  if (request.expected_outputs != plan->expected_outputs ||
      io == nullptr ||
      io->selected_token_out == nullptr ||
      io->selected_score_out == nullptr ||
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

template <emel::generator::attention_mode mode>
inline bool run_kernel_mode(const emel::graph::processor::event::execute & request,
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

  const bool ok = plan->kind == step_kind::prefill ? run_prefill<mode>(*backend)
                                                   : run_decode<mode>(*backend, request);
  if (err_out != nullptr) {
    *err_out = ok ? k_error_ok : k_error_invalid;
  }
  return ok;
}

template <emel::generator::attention_mode mode>
inline bool run_kernel_mode_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  const auto * plan = request_plan(request, err_out);
  if (plan == nullptr || !check_backend(backend, err_out) || !backend->bound_ready ||
      io == nullptr || io->selected_token_out == nullptr || io->selected_score_out == nullptr) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  const bool ok = plan->kind == step_kind::prefill
      ? run_prefill_preselected_argmax<mode>(*backend, *io->selected_token_out, *io->selected_score_out)
      : run_decode_preselected_argmax<mode>(
            *backend, request, *io->selected_token_out, *io->selected_score_out);
  if (err_out != nullptr) {
    *err_out = ok ? k_error_ok : k_error_invalid;
  }
  return ok;
}

inline bool run_kernel_flash(const emel::graph::processor::event::execute & request,
                             int32_t * err_out) noexcept {
  return run_kernel_mode<emel::generator::attention_mode::flash>(request, err_out);
}

inline bool run_kernel_nonflash(const emel::graph::processor::event::execute & request,
                                int32_t * err_out) noexcept {
  return run_kernel_mode<emel::generator::attention_mode::nonflash>(request, err_out);
}

template <emel::generator::attention_mode mode>
inline bool run_kernel_prefill_chunk4_mode(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  const auto * plan = request_plan(request, err_out);
  if (plan == nullptr || plan->kind != step_kind::prefill ||
      !check_backend(backend, err_out) || !backend->bound_ready ||
      backend->bound_token_count < k_prefill_q8_chunk_rows ||
      !prefill_chunk4_q8_gemm_supported(*backend)) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  const bool ok = run_prefill_chunk4<mode>(*backend);
  if (err_out != nullptr) {
    *err_out = ok ? k_error_ok : k_error_invalid;
  }
  return ok;
}

inline bool run_kernel_flash_prefill_chunk4(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<emel::generator::attention_mode::flash>(
      request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk4(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<emel::generator::attention_mode::nonflash>(
      request, err_out);
}

inline bool run_kernel_flash_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_mode_preselected_argmax<emel::generator::attention_mode::flash>(
      request, err_out);
}

inline bool run_kernel_nonflash_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_mode_preselected_argmax<emel::generator::attention_mode::nonflash>(
      request, err_out);
}

template <emel::generator::attention_mode mode>
inline bool run_kernel_prefill_chunk4_preselected_argmax_mode(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  const auto * plan = request_plan(request, err_out);
  if (plan == nullptr || plan->kind != step_kind::prefill ||
      !check_backend(backend, err_out) || !backend->bound_ready ||
      backend->bound_token_count < k_prefill_q8_chunk_rows ||
      !prefill_chunk4_q8_gemm_supported(*backend) ||
      io == nullptr || io->selected_token_out == nullptr || io->selected_score_out == nullptr) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
  }

  const bool ok = run_prefill_chunk4_preselected_argmax<mode>(
      *backend, *io->selected_token_out, *io->selected_score_out);
  if (err_out != nullptr) {
    *err_out = ok ? k_error_ok : k_error_invalid;
  }
  return ok;
}

inline bool run_kernel_flash_prefill_chunk4_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::generator::attention_mode::flash>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk4_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::generator::attention_mode::nonflash>(request, err_out);
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

inline bool extract_preselected_argmax(const emel::graph::processor::event::execute & request,
                                       int32_t * outputs_out,
                                       int32_t * err_out) noexcept {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<native_backend *>(io != nullptr ? io->backend_ctx : nullptr);
  if (!check_backend(backend, err_out) ||
      io == nullptr ||
      io->selected_token_out == nullptr ||
      io->selected_score_out == nullptr) {
    if (err_out != nullptr) {
      *err_out = k_error_invalid;
    }
    return false;
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
