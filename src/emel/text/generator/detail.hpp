#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "emel/text/generator/events.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/tensor/window/sm.hpp"

namespace emel::text::generator::detail {

struct tensor_matrix {
  const emel::model::data::tensor_record * tensor = nullptr;
  int32_t rows = 0;
  int32_t cols = 0;
};

struct rope_pairing {
  int32_t x0_stride = 2;
  int32_t x1_stride = 2;
  int32_t x1_offset = 1;
  int32_t x1_half_rot_offset = 0;
};

inline constexpr rope_pairing normal_rope_pairing() noexcept { return {}; }

inline constexpr rope_pairing neox_rope_pairing() noexcept {
  return rope_pairing{1, 1, 0, 1};
}

struct packed_matrix_binding {
  emel::model::data::tensor_record tensor = {};
  std::vector<uint8_t> storage = {};
  // The raw model record the packed layout was prepared from. Streamed slots
  // hold raw GGUF bytes, so the streamed rebase must clone this record (its
  // dtype/extents describe the slot bytes) while resident restore keeps the
  // packed record the layout swap bound.
  const emel::model::data::tensor_record * source = nullptr;
};

struct block_weights {
  bool uses_attention = true;
  int32_t attention_q_dim = 0;
  int32_t attention_kv_dim = 0;
  int32_t attention_head_dim = 0;
  int32_t attention_head_dim_kv = 0;
  int32_t attention_rope_dim = 0;
  float attention_rope_freq_base = 10000.0f;
  rope_pairing attention_rope_pairing = {};
  std::vector<float> attention_norm = {};
  tensor_matrix attention_q = {};
  packed_matrix_binding attention_q_packed = {};
  tensor_matrix attention_k = {};
  packed_matrix_binding attention_k_packed = {};
  tensor_matrix attention_v = {};
  packed_matrix_binding attention_v_packed = {};
  std::vector<float> attention_q_norm = {};
  std::vector<float> attention_k_norm = {};
  tensor_matrix attention_output = {};
  packed_matrix_binding attention_output_packed = {};
  std::vector<float> shortconv_conv = {};
  tensor_matrix shortconv_in_proj = {};
  packed_matrix_binding shortconv_in_proj_packed = {};
  tensor_matrix shortconv_out_proj = {};
  packed_matrix_binding shortconv_out_proj_packed = {};
  std::vector<float> feed_forward_norm = {};
  tensor_matrix feed_forward_gate = {};
  packed_matrix_binding feed_forward_gate_packed = {};
  tensor_matrix feed_forward_down = {};
  packed_matrix_binding feed_forward_down_packed = {};
  tensor_matrix feed_forward_up = {};
  packed_matrix_binding feed_forward_up_packed = {};
};

// View-sliced parallel matmul lanes: one kernel actor per pool worker. A
// parallel dispatch forks one logical matmul into per-lane row-slice events
// and joins before the enclosing action returns, so no work escapes the RTC
// boundary and no lane actor is entered concurrently.
constexpr size_t k_matmul_lanes = 8;
using matmul_lane_pool = emel::policy::thread_pool_scheduler<k_matmul_lanes, 16u, 128u>;
using matmul_lane_scheduler = emel::policy::thread_pool_scheduler_ref<matmul_lane_pool>;

enum class matmul_lane_mode : uint8_t {
  serial = 0,
  parallel = 1,
};

// Weight residency mode for the layer loop: resident consumes blocks[] views
// as prepared; streamed acquires each layer's slot from the tensor window
// actor and rebases the block's matmul weight views into the slot before use.
enum class window_mode : uint8_t {
  resident = 0,
  streamed = 1,
};

// Canonical per-layer stream role order shared by extent builders (tests,
// bench fixtures) and the rebase walk below. Absent roles are skipped; both
// sides must walk this exact order for positional extents to line up.
inline constexpr size_t k_stream_role_attention_q = 0;
inline constexpr size_t k_stream_role_attention_k = 1;
inline constexpr size_t k_stream_role_attention_v = 2;
inline constexpr size_t k_stream_role_attention_output = 3;
inline constexpr size_t k_stream_role_feed_forward_gate = 4;
inline constexpr size_t k_stream_role_feed_forward_down = 5;
inline constexpr size_t k_stream_role_feed_forward_up = 6;
inline constexpr size_t k_stream_role_shortconv_in_proj = 7;
inline constexpr size_t k_stream_role_shortconv_out_proj = 8;
inline constexpr size_t k_stream_role_count = 9;

// Streamed-weight binding: injected window actor plus the per-role record
// clones the layer loop rebases blocks[] onto. One clone set suffices because
// single-lane decode consumes exactly one layer at a time. pristine holds the
// prepare()-time record pointers per layer so every rebase clones from the
// original model contract (blocks[] pointers move onto the clones after the
// first streamed pass).
struct stream_binding {
  emel::model::tensor::window::sm * window = nullptr;
  bool active = false;
  std::array<emel::model::data::tensor_record, k_stream_role_count> records = {};
  // pristine holds the prepare()-time bound records (packed on aarch64) that
  // resident steps restore to; raw holds the untouched model records whose
  // dtype/extents describe the raw GGUF bytes the window streams into slots.
  std::vector<std::array<const emel::model::data::tensor_record *, k_stream_role_count>>
      pristine = {};
  std::vector<std::array<const emel::model::data::tensor_record *, k_stream_role_count>>
      raw = {};
  // The logits/argmax output stage of a streamed step runs on the raw model
  // record too (a packed resident output would need the packed input
  // pipeline, which the raw-classified streamed route does not run); resident
  // steps restore the prepare()-time views.
  const emel::model::data::tensor_record * pristine_output = nullptr;
  const emel::model::data::tensor_record * raw_output = nullptr;
  const emel::model::data::tensor_record * pristine_output_argmax = nullptr;
  const emel::model::data::tensor_record * raw_output_argmax = nullptr;
};

struct native_backend {
  const emel::model::data * model = nullptr;
  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan prefill_plan = {};
  emel::model::llama::detail::step_plan decode_plan = {};
  emel::kernel::sm kernel = {};
  // Parallel matmul lane actors and their pool. Worker threads are one-time
  // prepare() construction (in-place, no heap); slice dispatch reuses these
  // actors and allocates nothing.
  std::array<emel::kernel::sm, k_matmul_lanes> lane_kernels = {};
  std::optional<matmul_lane_pool> lane_pool = {};
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
  std::vector<emel::kernel::detail::quant::block_q8_k> q8_input_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_k> q8_input_chunk4_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_k> q8_input_chunk8_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_0> packed_q8_0_input_storage = {};
  std::vector<emel::kernel::detail::quant::block_q8_0> packed_q8_0_chunk4_rows = {};
  std::vector<uint8_t> packed_q8_0_chunk4_input_storage = {};
  emel::model::llama::detail::quantized_path_audit quantized_audit = {};
  std::vector<block_weights> blocks = {};
  stream_binding stream = {};

  int32_t n_vocab = 0;
  int32_t n_embd = 0;
  int32_t n_head = 0;
  int32_t n_head_kv = 0;
  int32_t n_layer = 0;
  int32_t n_ctx = 0;
  int32_t n_rot = 0;
  int32_t head_dim = 0;
  int32_t head_dim_kv = 0;
  int32_t max_q_dim = 0;
  int32_t max_kv_dim = 0;
  int32_t max_ffn_dim = 0;
  int32_t n_rep = 0;
  int32_t shortconv_kernel_size = 0;
  int32_t shortconv_state_size = 0;
  float rms_epsilon = 1.0e-5f;
  float rope_freq_base = 10000.0f;

  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<uint16_t> flash_key_cache = {};
  std::vector<uint16_t> flash_value_cache = {};
  std::vector<size_t> layer_cache_offsets = {};
  std::vector<size_t> flash_layer_cache_offsets = {};
  std::vector<float> recurrent_shortconv_cache = {};
  int32_t kv_cache_tokens = 0;
  // Physical KV geometry from the memory-domain contract (emel::memory::view):
  // per-layer position capacity is n_ctx rounded up to whole blocks so the
  // block map and the physical layout agree on extents.
  int32_t kv_block_tokens = 0;
  int32_t kv_positions_capacity = 0;

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
  std::vector<float> hidden_chunk8 = {};
  std::vector<float> norm = {};
  std::vector<float> norm_chunk4 = {};
  std::vector<float> norm_chunk8 = {};
  std::vector<float> shortconv_bcx = {};
  std::vector<float> shortconv_bx = {};
  std::vector<float> shortconv_conv_out = {};
  std::vector<float> shortconv_bcx_chunk4 = {};
  std::vector<float> shortconv_conv_out_chunk4 = {};
  std::vector<float> shortconv_bcx_chunk8 = {};
  std::vector<float> shortconv_conv_out_chunk8 = {};
  std::vector<float> q = {};
  std::vector<float> q_attn = {};
  std::vector<float> q_chunk4 = {};
  std::vector<float> q_chunk8 = {};
  std::vector<float> k = {};
  std::vector<float> k_chunk4 = {};
  std::vector<float> k_chunk8 = {};
  std::vector<float> v = {};
  std::vector<float> v_chunk4 = {};
  std::vector<float> v_chunk8 = {};
  std::vector<float> attn_scores = {};
  std::vector<float> attn_probs = {};
  std::vector<float> attn_probs_rounded = {};
  std::vector<float> attn_value_column = {};
  std::vector<float> attn_ctx = {};
  std::vector<float> attn_ctx_chunk4 = {};
  std::vector<float> attn_ctx_chunk8 = {};
  std::vector<float> projected = {};
  std::vector<float> projected_chunk4 = {};
  std::vector<float> projected_chunk8 = {};
  std::vector<float> gate = {};
  std::vector<float> gate_chunk4 = {};
  std::vector<float> gate_chunk8 = {};
  std::vector<float> up = {};
  std::vector<float> up_chunk4 = {};
  std::vector<float> up_chunk8 = {};
  std::vector<float> ffn_hidden = {};
  std::vector<float> ffn_hidden_chunk4 = {};
  std::vector<float> ffn_hidden_chunk8 = {};
  bool bound_ready = false;
};

namespace quant = emel::kernel::detail::quant;

namespace {

using tensor_record = emel::model::data::tensor_record;
using step_kind = emel::model::llama::detail::step_kind;

enum class chunk4_rhs_route : uint8_t {
  packed_q8_0 = 0,
  q8_k = 1,
};

enum class scalar_matmul_route : uint8_t {
  packed_q8_0 = 0,
  q8_k = 1,
  native_quantized = 2,
  native_quantized_q8_k_logits = 3,
  kernel = 4,
};

enum class scalar_argmax_route : uint8_t {
  q8_k = 0,
  kernel = 1,
};
using step_plan = emel::model::llama::detail::step_plan;

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_invalid = 1;
// A streamed decode could not acquire its layer window slot: distinct from
// generic compute failure so the outcome is externally attributable to the
// tensor-window collaborator.
constexpr int32_t k_error_stream_acquire = 2;
constexpr int32_t k_prefill_q8_chunk_rows = 4;
constexpr int32_t k_prefill_q8_chunk8_rows = 8;
// Minimum prompt size before the parallel matmul lanes are worth the fork
// overhead; route guards read this so the choice stays in transition rows.
constexpr int32_t k_parallel_min_prefill_tokens = 8;
// Decode GEMV lanes only pay off once per-matmul work dwarfs the fork
// overhead; route guards require this minimum model width for parallel decode.
constexpr int32_t k_parallel_min_gemv_dim = 1024;
constexpr emel::kernel::kernel_kind detect_host_kernel_kind() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
  return emel::kernel::kernel_kind::aarch64;
#elif defined(__x86_64__) || defined(_M_X64)
  return emel::kernel::kernel_kind::x86_64;
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

inline emel::kernel::event::tensor_view make_q8_k_rhs_chunk4_view(
    const emel::kernel::detail::quant::block_q8_k * data,
    const uint64_t cols) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const size_t row_bytes =
      emel::kernel::detail::quantized_row_storage_bytes(
          emel::kernel::detail::dtype_q8_k, cols);
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::q8_k_x4;
  tensor.ne = {
      static_cast<uint64_t>(k_prefill_q8_chunk_rows),
      cols,
      1u,
      1u,
  };
  tensor.nb[0] = 1u;
  tensor.nb[1] = row_bytes;
  tensor.nb[2] = row_bytes * static_cast<uint64_t>(k_prefill_q8_chunk_rows);
  tensor.nb[3] = tensor.nb[2];
  return tensor;
}

inline emel::kernel::event::tensor_view make_q8_k_rhs_chunk8_view(
    const emel::kernel::detail::quant::block_q8_k * data,
    const uint64_t cols) noexcept {
  emel::kernel::event::tensor_view tensor{};
  const size_t row_bytes =
      emel::kernel::detail::quantized_row_storage_bytes(
          emel::kernel::detail::dtype_q8_k, cols);
  tensor.data = data;
  tensor.type = emel::kernel::event::dtype::q8_k_x8;
  tensor.ne = {
      static_cast<uint64_t>(k_prefill_q8_chunk8_rows),
      cols,
      1u,
      1u,
  };
  tensor.nb[0] = 1u;
  tensor.nb[1] = row_bytes;
  tensor.nb[2] = row_bytes * static_cast<uint64_t>(k_prefill_q8_chunk8_rows);
  tensor.nb[3] = tensor.nb[2];
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
  return emel::kernel::detail::row_storage_bytes_for_dtype(
      static_cast<uint8_t>(tensor.type),
      static_cast<uint64_t>(cols));
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

inline bool is_lfm2_runtime(const native_backend & backend) noexcept;
inline bool q8_input_path_supported(const native_backend & backend,
                                    const tensor_matrix & matrix) noexcept;

inline bool q8_input_chunk4_path_supported(const native_backend & backend,
                                           const tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.q8_input_chunk4_storage.empty()) {
    return false;
  }
  return q8_input_path_supported(backend, matrix);
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool q8_input_chunk8_path_supported(const native_backend & backend,
                                           const tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.q8_input_chunk8_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  return dtype == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared;
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline int32_t effective_attention_q_dim(const native_backend & backend,
                                         const block_weights & block) noexcept;
inline int32_t effective_attention_kv_dim(const native_backend & backend,
                                          const block_weights & block) noexcept;
inline int32_t effective_attention_head_dim(const native_backend & backend,
                                            const block_weights & block) noexcept;
inline int32_t effective_attention_head_dim_kv(const native_backend & backend,
                                               const block_weights & block) noexcept;
inline int32_t effective_attention_rope_dim(const native_backend & backend,
                                            const block_weights & block) noexcept;
inline float effective_attention_rope_freq_base(const native_backend & backend,
                                                const block_weights & block) noexcept;
inline int32_t effective_max_q_dim(const native_backend & backend) noexcept;
inline int32_t effective_max_kv_dim(const native_backend & backend) noexcept;
inline int32_t effective_max_ffn_dim(const native_backend & backend) noexcept;

template <chunk4_rhs_route route>
inline bool chunk4_matmul_backend_ready(const native_backend & backend,
                                        const tensor_matrix & matrix) noexcept {
  if constexpr (route == chunk4_rhs_route::packed_q8_0) {
    return packed_q8_0_chunk4_input_path_supported(backend, matrix);
  } else {
    return q8_input_chunk4_path_supported(backend, matrix);
  }
}

template <chunk4_rhs_route route>
inline bool prefill_chunk4_backend_ready(const native_backend & backend) noexcept {
  const int32_t max_q_dim = effective_max_q_dim(backend);
  const int32_t max_kv_dim = effective_max_kv_dim(backend);
  const int32_t max_ffn_dim = effective_max_ffn_dim(backend);
  if (backend.blocks.empty() || backend.n_layer <= 0) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? chunk4_matmul_backend_ready<route>(backend, block.attention_q) &&
              chunk4_matmul_backend_ready<route>(backend, block.attention_k) &&
              chunk4_matmul_backend_ready<route>(backend, block.attention_v) &&
              chunk4_matmul_backend_ready<route>(backend, block.attention_output)
        : backend.shortconv_kernel_size > 1 &&
              chunk4_matmul_backend_ready<route>(backend, block.shortconv_in_proj) &&
              chunk4_matmul_backend_ready<route>(backend, block.shortconv_out_proj) &&
              !block.shortconv_conv.empty();
    if (!residual_ok ||
        !chunk4_matmul_backend_ready<route>(backend, block.feed_forward_gate) ||
        !chunk4_matmul_backend_ready<route>(backend, block.feed_forward_down) ||
        !chunk4_matmul_backend_ready<route>(backend, block.feed_forward_up)) {
      return false;
    }
  }

  const bool shortconv_ok = backend.shortconv_state_size == 0 ||
      (backend.shortconv_bcx_chunk4.size() ==
           static_cast<size_t>(k_prefill_q8_chunk_rows) *
               static_cast<size_t>(3 * backend.n_embd) &&
       backend.shortconv_conv_out_chunk4.size() == backend.hidden_chunk4.size());
  return shortconv_ok &&
      backend.hidden_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(backend.n_embd) &&
      backend.norm_chunk4.size() == backend.hidden_chunk4.size() &&
      backend.projected_chunk4.size() == backend.hidden_chunk4.size() &&
      backend.attn_ctx_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(max_q_dim) &&
      backend.q_chunk4.size() == backend.attn_ctx_chunk4.size() &&
      backend.k_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(max_kv_dim) &&
      backend.v_chunk4.size() == backend.k_chunk4.size() &&
      backend.gate_chunk4.size() ==
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(max_ffn_dim) &&
      backend.up_chunk4.size() == backend.gate_chunk4.size() &&
      backend.ffn_hidden_chunk4.size() == backend.gate_chunk4.size();
}

inline bool prefill_chunk4_q8_gemm_backend_ready(const native_backend & backend) noexcept {
  return prefill_chunk4_backend_ready<chunk4_rhs_route::packed_q8_0>(backend) ||
      prefill_chunk4_backend_ready<chunk4_rhs_route::q8_k>(backend);
}

inline bool prefill_chunk8_q8_k_backend_ready(const native_backend & backend) noexcept {
  const int32_t max_q_dim = effective_max_q_dim(backend);
  const int32_t max_kv_dim = effective_max_kv_dim(backend);
  const int32_t max_ffn_dim = effective_max_ffn_dim(backend);
  if (backend.blocks.empty() || backend.n_layer <= 0) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? q8_input_chunk8_path_supported(backend, block.attention_q) &&
              q8_input_chunk8_path_supported(backend, block.attention_k) &&
              q8_input_chunk8_path_supported(backend, block.attention_v) &&
              q8_input_chunk8_path_supported(backend, block.attention_output)
        : backend.shortconv_kernel_size > 1 &&
              q8_input_chunk8_path_supported(backend, block.shortconv_in_proj) &&
              q8_input_chunk8_path_supported(backend, block.shortconv_out_proj) &&
              !block.shortconv_conv.empty();
    if (!residual_ok ||
        !q8_input_chunk8_path_supported(backend, block.feed_forward_gate) ||
        !q8_input_chunk8_path_supported(backend, block.feed_forward_down) ||
        !q8_input_chunk8_path_supported(backend, block.feed_forward_up)) {
      return false;
    }
  }

  const bool shortconv_ok = backend.shortconv_state_size == 0 ||
      (backend.shortconv_bcx_chunk8.size() ==
           static_cast<size_t>(k_prefill_q8_chunk8_rows) *
               static_cast<size_t>(3 * backend.n_embd) &&
       backend.shortconv_conv_out_chunk8.size() == backend.hidden_chunk8.size());
  return shortconv_ok &&
      backend.hidden_chunk8.size() ==
          static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(backend.n_embd) &&
      backend.norm_chunk8.size() == backend.hidden_chunk8.size() &&
      backend.projected_chunk8.size() == backend.hidden_chunk8.size() &&
      backend.attn_ctx_chunk8.size() ==
          static_cast<size_t>(k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(max_q_dim) &&
      backend.q_chunk8.size() == backend.attn_ctx_chunk8.size() &&
      backend.k_chunk8.size() ==
          static_cast<size_t>(k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(max_kv_dim) &&
      backend.v_chunk8.size() == backend.k_chunk8.size() &&
      backend.gate_chunk8.size() ==
          static_cast<size_t>(k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(max_ffn_dim) &&
      backend.up_chunk8.size() == backend.gate_chunk8.size() &&
      backend.ffn_hidden_chunk8.size() == backend.gate_chunk8.size();
}

inline bool q8_input_workspace_candidate(const tensor_matrix & matrix) noexcept {
  if (matrix.tensor == nullptr) {
    return false;
  }
  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  return dtype == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
      dtype == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
}

inline bool q8_input_path_supported(const native_backend & backend,
                                    const tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.q8_input_storage.empty() ||
      matrix.tensor == nullptr) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (dtype == emel::kernel::detail::dtype_q4_k_x8_bl8) {
    return true;
  }
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared) {
    return true;
  }
#endif
#if defined(__ARM_FEATURE_DOTPROD)
  if (dtype == emel::kernel::detail::dtype_q4_k_x8_bl4) {
    return true;
  }
  if (dtype == emel::kernel::detail::dtype_q6_k_x8) {
    return true;
  }
#endif
  return false;
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool q8_input_argmax_path_supported(const native_backend & backend,
                                           const tensor_matrix & matrix) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (q8_input_path_supported(backend, matrix)) {
    return true;
  }
  return backend.kernel_kind == emel::kernel::kernel_kind::aarch64 &&
      !backend.q8_input_storage.empty() &&
      matrix.tensor != nullptr &&
#if defined(__ARM_FEATURE_MATMUL_INT8)
      static_cast<uint8_t>(matrix.tensor->type) ==
          emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
#else
      false;
#endif
#else
  (void) backend;
  (void) matrix;
  return false;
#endif
}

inline bool preselected_argmax_direct_supported(const native_backend & backend) noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON)
  if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64 ||
      backend.output_argmax.tensor == nullptr ||
      backend.q8_input_storage.empty()) {
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
    case emel::kernel::event::dtype::q4_k:
      quant::dequantize_row_q4_k(reinterpret_cast<const quant::block_q4_k *>(src_row),
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

inline bool is_gemma4_runtime(const native_backend & backend) noexcept {
  return backend.model != nullptr && emel::model::architecture_name_view(*backend.model) == "gemma4";
}

inline bool is_gemma4_sliding_attention_layer(const native_backend & backend,
                                              const int32_t layer_index) noexcept {
  return is_gemma4_runtime(backend) &&
         layer_index >= 0 &&
         static_cast<uint32_t>(layer_index) <
             backend.model->params.attention_sliding_window_pattern_count &&
         backend.model->params.attention_sliding_window_pattern_flags[
             static_cast<size_t>(layer_index)] != 0u;
}

inline bool is_gemma4_shared_kv_layer(const native_backend & backend,
                                      const int32_t layer_index) noexcept {
  return is_gemma4_runtime(backend) &&
      layer_index >= 0 &&
      backend.model != nullptr &&
      backend.model->params.attention_shared_kv_layers > 0 &&
      layer_index >= (backend.n_layer - backend.model->params.attention_shared_kv_layers) &&
      layer_index < backend.n_layer;
}

inline bool is_lfm2_runtime(const native_backend & backend) noexcept {
  return backend.model != nullptr && emel::model::architecture_name_view(*backend.model) == "lfm2";
}

inline bool requires_attention_qk_norm(const native_backend & backend,
                                       const block_weights & block) noexcept {
  return block.uses_attention &&
      (is_qwen3_runtime(backend) || is_gemma4_runtime(backend) || is_lfm2_runtime(backend));
}

inline int32_t effective_attention_q_dim(const native_backend & backend,
                                         const block_weights & block) noexcept {
  if (block.attention_q_dim > 0) {
    return block.attention_q_dim;
  }
  if (block.attention_q.rows > 0) {
    return block.attention_q.rows;
  }
  return backend.n_head > 0 ? backend.n_head * backend.head_dim : 0;
}

inline int32_t effective_attention_kv_dim(const native_backend & backend,
                                          const block_weights & block) noexcept {
  if (block.attention_kv_dim > 0) {
    return block.attention_kv_dim;
  }
  if (block.attention_k.rows > 0) {
    return block.attention_k.rows;
  }
  return backend.n_head_kv > 0 ? backend.n_head_kv * backend.head_dim_kv : 0;
}

inline int32_t effective_attention_head_dim(const native_backend & backend,
                                            const block_weights & block) noexcept {
  if (block.attention_head_dim > 0) {
    return block.attention_head_dim;
  }
  if (block.attention_q.rows > 0 && backend.n_head > 0) {
    return block.attention_q.rows / backend.n_head;
  }
  return backend.head_dim;
}

inline int32_t effective_attention_head_dim_kv(const native_backend & backend,
                                               const block_weights & block) noexcept {
  if (block.attention_head_dim_kv > 0) {
    return block.attention_head_dim_kv;
  }
  if (block.attention_k.rows > 0 && backend.n_head_kv > 0) {
    return block.attention_k.rows / backend.n_head_kv;
  }
  return backend.head_dim_kv;
}

inline int32_t effective_attention_rope_dim(const native_backend & backend,
                                            const block_weights & block) noexcept {
  if (block.attention_rope_dim > 0) {
    return block.attention_rope_dim;
  }
  return backend.n_rot;
}

inline float effective_attention_rope_freq_base(const native_backend & backend,
                                                const block_weights & block) noexcept {
  if (block.attention_rope_dim > 0) {
    return block.attention_rope_freq_base;
  }
  return backend.rope_freq_base > 0.0f ? backend.rope_freq_base : block.attention_rope_freq_base;
}

inline int32_t effective_max_q_dim(const native_backend & backend) noexcept {
  return backend.max_q_dim > 0 ? backend.max_q_dim : backend.n_head * backend.head_dim;
}

inline int32_t effective_max_kv_dim(const native_backend & backend) noexcept {
  return backend.max_kv_dim > 0 ? backend.max_kv_dim : backend.n_head_kv * backend.head_dim_kv;
}

inline int32_t effective_max_ffn_dim(const native_backend & backend) noexcept {
  return backend.max_ffn_dim > 0
      ? backend.max_ffn_dim
      : (backend.blocks.empty() ? 0 : backend.blocks.front().feed_forward_gate.rows);
}

inline const block_weights * first_attention_block(const native_backend & backend) noexcept {
  for (const auto & block : backend.blocks) {
    if (block.uses_attention) {
      return &block;
    }
  }

  return nullptr;
}

struct kv_addressing_view {
  const uint16_t * blocks = nullptr;
  int32_t block_tokens = 1;
  int32_t recurrent_slot = 0;
};

inline constexpr std::array<uint16_t, emel::memory::view::MAX_BLOCKS_PER_SEQUENCE>
make_identity_kv_blocks() noexcept {
  std::array<uint16_t, emel::memory::view::MAX_BLOCKS_PER_SEQUENCE> blocks = {};
  for (size_t idx = 0; idx < blocks.size(); ++idx) {
    blocks[idx] = static_cast<uint16_t>(idx);
  }
  return blocks;
}

inline constexpr auto k_identity_kv_blocks = make_identity_kv_blocks();

inline kv_addressing_view identity_kv_addressing() noexcept {
  return kv_addressing_view{
    .blocks = k_identity_kv_blocks.data(),
    .block_tokens = 1,
    .recurrent_slot = 0,
  };
}

inline kv_addressing_view kv_addressing_from_snapshot(
    const emel::memory::view::snapshot & snapshot, const int32_t seq_id) noexcept {
  return kv_addressing_view{
    .blocks = snapshot.sequence_kv_blocks[static_cast<size_t>(seq_id)].data(),
    .block_tokens = snapshot.block_tokens,
    .recurrent_slot = snapshot.lookup_recurrent_slot(seq_id),
  };
}

inline kv_addressing_view kv_addressing_from_request(
    const emel::graph::processor::event::execute & request) noexcept {
  return kv_addressing_from_snapshot(*request.memory_view, request.seq_primary_ids[0]);
}

inline size_t shortconv_state_layer_offset(const native_backend & backend,
                                           const kv_addressing_view & kv,
                                           const int32_t layer_index) noexcept {
  // Recurrent state is addressed through the snapshot-resolved slot bound at
  // compute dispatch; the cache holds max_sequences (currently 1) slots, so
  // slot 0 preserves the flat pre-cutover layout bit-exactly.
  return (static_cast<size_t>(kv.recurrent_slot) *
              static_cast<size_t>(backend.n_layer) +
          static_cast<size_t>(layer_index)) *
         static_cast<size_t>(backend.shortconv_state_size) *
         static_cast<size_t>(backend.n_embd);
}

inline size_t shortconv_state_layer_offset(const native_backend & backend,
                                           const int32_t layer_index) noexcept {
  return shortconv_state_layer_offset(backend, identity_kv_addressing(), layer_index);
}

inline void reset_shortconv_cache(native_backend & backend) noexcept {
  std::fill(
      backend.recurrent_shortconv_cache.begin(), backend.recurrent_shortconv_cache.end(), 0.0f);
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

inline bool apply_rms_norm_in_place(std::span<float> vector, const float epsilon) noexcept {
  if (vector.empty()) {
    return false;
  }

  double square_sum = 0.0;
  for (const float value : vector) {
    square_sum += static_cast<double>(value * value);
  }

  const float mean = static_cast<float>(square_sum / static_cast<double>(vector.size()));
  const float scale = 1.0f / std::sqrt(mean + epsilon);
  for (float & value : vector) {
    value *= scale;
  }

  return true;
}

inline bool apply_attention_qk_norm(native_backend & backend,
                                    const block_weights & block) noexcept {
  auto q = std::span<float>(
      backend.q.data(), static_cast<size_t>(effective_attention_q_dim(backend, block)));
  auto k = std::span<float>(
      backend.k.data(), static_cast<size_t>(effective_attention_kv_dim(backend, block)));
  return apply_headwise_rms_norm(
             q,
             block.attention_q_norm,
             backend.n_head,
             effective_attention_head_dim(backend, block),
             backend.rms_epsilon) &&
      apply_headwise_rms_norm(
             k,
             block.attention_k_norm,
             backend.n_head_kv,
             effective_attention_head_dim_kv(backend, block),
             backend.rms_epsilon);
}

inline bool apply_qwen3_attention_qk_norm(native_backend & backend,
                                          const block_weights & block) noexcept {
  return apply_attention_qk_norm(backend, block);
}

inline bool requires_attention_v_norm(const native_backend & backend,
                                      const int32_t layer_index,
                                      const block_weights & block) noexcept {
  return block.uses_attention && is_gemma4_shared_kv_layer(backend, layer_index);
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
                                              packed_matrix_binding & packed) noexcept {
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
  packed.source = matrix.tensor;
  matrix.tensor = &packed.tensor;
  return true;
}

template <uint8_t packed_dtype>
inline bool prepare_packed_q4_tensor_layout(const tensor_record & source,
                                            const int32_t rows,
                                            const int32_t cols,
                                            tensor_record & packed_tensor,
                                            std::vector<uint8_t> & packed_storage) noexcept {
  if (source.data == nullptr ||
      static_cast<uint8_t>(source.type) != emel::kernel::detail::dtype_q4_k ||
      rows <= 0 ||
      cols <= 0) {
    return false;
  }

  const uint64_t urows = static_cast<uint64_t>(rows);
  const uint64_t ucols = static_cast<uint64_t>(cols);
  const uint64_t group_count = emel::kernel::detail::quant::packed_q4_k_x8_group_count(urows);
  const size_t group_bytes =
      emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(ucols);
  const uint64_t storage_bytes = static_cast<uint64_t>(group_bytes) * group_count;
  if (group_bytes == 0u || storage_bytes == 0u) {
    return false;
  }

  packed_storage.resize(static_cast<size_t>(storage_bytes));
  const auto * src =
      reinterpret_cast<const emel::kernel::detail::quant::block_q4_k *>(source.data);
  bool packed_ok = false;
  if constexpr (packed_dtype == emel::kernel::detail::dtype_q4_k_x8_bl8) {
    packed_ok = emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
        src, urows, ucols, packed_storage.data());
  } else if constexpr (packed_dtype == emel::kernel::detail::dtype_q4_k_x8_bl4) {
    packed_ok = emel::kernel::detail::quant::pack_q4_k_rows_x8_bl4(
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
inline bool prepare_packed_q4_matrix_layout(tensor_matrix & matrix,
                                            packed_matrix_binding & packed) noexcept {
  if (matrix.tensor == nullptr) {
    return false;
  }
  if (static_cast<uint8_t>(matrix.tensor->type) != emel::kernel::detail::dtype_q4_k) {
    return true;
  }
  if (!prepare_packed_q4_tensor_layout<packed_dtype>(
          *matrix.tensor, matrix.rows, matrix.cols, packed.tensor, packed.storage)) {
    return false;
  }
  packed.source = matrix.tensor;
  matrix.tensor = &packed.tensor;
  return true;
}

template <uint8_t packed_dtype>
inline bool prepare_packed_q6_tensor_layout(const tensor_record & source,
                                            const int32_t rows,
                                            const int32_t cols,
                                            tensor_record & packed_tensor,
                                            std::vector<uint8_t> & packed_storage) noexcept {
  if (source.data == nullptr ||
      static_cast<uint8_t>(source.type) != emel::kernel::detail::dtype_q6_k ||
      rows <= 0 ||
      cols <= 0) {
    return false;
  }

  const uint64_t urows = static_cast<uint64_t>(rows);
  const uint64_t ucols = static_cast<uint64_t>(cols);
  const uint64_t group_count = emel::kernel::detail::quant::packed_q6_k_x8_group_count(urows);
  const size_t group_bytes =
      packed_dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared
      ? emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(ucols)
      : emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(ucols);
  const uint64_t storage_bytes = static_cast<uint64_t>(group_bytes) * group_count;
  if (group_bytes == 0u || storage_bytes == 0u) {
    return false;
  }

  packed_storage.resize(static_cast<size_t>(storage_bytes));
  const auto * src =
      reinterpret_cast<const emel::kernel::detail::quant::block_q6_k *>(source.data);
  bool packed_ok = false;
  if constexpr (packed_dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared) {
    packed_ok = emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_prepared(
        src, urows, ucols, packed_storage.data());
  } else if constexpr (packed_dtype == emel::kernel::detail::dtype_q6_k_x8) {
    packed_ok = emel::kernel::detail::quant::pack_q6_k_rows_x8(
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
inline bool prepare_packed_q6_matrix_layout(tensor_matrix & matrix,
                                            packed_matrix_binding & packed) noexcept {
  if (matrix.tensor == nullptr) {
    return false;
  }
  if (static_cast<uint8_t>(matrix.tensor->type) != emel::kernel::detail::dtype_q6_k) {
    return true;
  }
  if (!prepare_packed_q6_tensor_layout<packed_dtype>(
          *matrix.tensor, matrix.rows, matrix.cols, packed.tensor, packed.storage)) {
    return false;
  }
  packed.source = matrix.tensor;
  matrix.tensor = &packed.tensor;
  return true;
}

inline bool prepare_native_matrix_layout(native_backend & backend,
                                         tensor_matrix & matrix,
                                         packed_matrix_binding & packed) noexcept {
  (void)packed;
  if (matrix.tensor == nullptr) {
    return false;
  }

  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  if (dtype == emel::kernel::detail::dtype_q4_k) {
    if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64) {
      return true;
    }
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    return prepare_packed_q4_matrix_layout<emel::kernel::detail::dtype_q4_k_x8_bl8>(
        matrix, packed);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return prepare_packed_q4_matrix_layout<emel::kernel::detail::dtype_q4_k_x8_bl4>(
        matrix, packed);
#else
    return true;
#endif
  }

  if (dtype == emel::kernel::detail::dtype_q6_k) {
    if (backend.kernel_kind != emel::kernel::kernel_kind::aarch64) {
      return true;
    }
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    return prepare_packed_q6_matrix_layout<emel::kernel::detail::dtype_q6_k_x8_q8_prepared>(
        matrix, packed);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return prepare_packed_q6_matrix_layout<emel::kernel::detail::dtype_q6_k_x8>(matrix, packed);
#else
    return true;
#endif
  }

  if (dtype != emel::kernel::detail::dtype_q8_0) {
    return true;
  }

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  return prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl8>(
      matrix, packed);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return prepare_packed_q8_0_matrix_layout<emel::kernel::detail::dtype_q8_0_x4_bl4>(
      matrix, packed);
#else
  (void) backend;
  return true;
#endif
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

inline bool prepare_block_native_matrices(native_backend & backend) noexcept {
  for (auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? prepare_native_matrix_layout(backend, block.attention_q, block.attention_q_packed) &&
              prepare_native_matrix_layout(backend, block.attention_k, block.attention_k_packed) &&
              prepare_native_matrix_layout(backend, block.attention_v, block.attention_v_packed) &&
              prepare_native_matrix_layout(
                  backend, block.attention_output, block.attention_output_packed)
        : prepare_native_matrix_layout(
              backend, block.shortconv_in_proj, block.shortconv_in_proj_packed) &&
              prepare_native_matrix_layout(
                  backend, block.shortconv_out_proj, block.shortconv_out_proj_packed);
    if (!residual_ok ||
        !prepare_native_matrix_layout(
            backend, block.feed_forward_gate, block.feed_forward_gate_packed) ||
        !prepare_native_matrix_layout(
            backend, block.feed_forward_down, block.feed_forward_down_packed) ||
        !prepare_native_matrix_layout(
            backend, block.feed_forward_up, block.feed_forward_up_packed)) {
      return false;
    }
  }
  return true;
}

inline bool update_q8_input_requirement(const tensor_matrix & matrix,
                                        size_t & max_block_count) noexcept {
  if (!q8_input_workspace_candidate(matrix)) {
    return true;
  }

  if (matrix.cols <= 0 ||
      (matrix.cols % static_cast<int32_t>(quant::QK_K)) != 0) {
    return false;
  }

  const size_t block_count =
      static_cast<size_t>(matrix.cols) / static_cast<size_t>(quant::QK_K);
  if (block_count == 0u || block_count > quant::MAX_Q8_K_BLOCKS) {
    return false;
  }

  max_block_count = std::max(max_block_count, block_count);
  return true;
}

inline bool prepare_q8_input_workspace(native_backend & backend) noexcept {
  backend.q8_input_storage.clear();

  size_t max_block_count = 0u;
  if (!update_q8_input_requirement(backend.output, max_block_count) ||
      !update_q8_input_requirement(backend.output_argmax, max_block_count)) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? update_q8_input_requirement(block.attention_q, max_block_count) &&
              update_q8_input_requirement(block.attention_k, max_block_count) &&
              update_q8_input_requirement(block.attention_v, max_block_count) &&
              update_q8_input_requirement(block.attention_output, max_block_count)
        : update_q8_input_requirement(block.shortconv_in_proj, max_block_count) &&
              update_q8_input_requirement(block.shortconv_out_proj, max_block_count);
    if (!residual_ok ||
        !update_q8_input_requirement(block.feed_forward_gate, max_block_count) ||
        !update_q8_input_requirement(block.feed_forward_down, max_block_count) ||
        !update_q8_input_requirement(block.feed_forward_up, max_block_count)) {
      return false;
    }
  }

  if (max_block_count == 0u) {
    return true;
  }

  backend.q8_input_storage.resize(max_block_count);
  return true;
}

inline bool prepare_q8_input_chunk4_workspace(native_backend & backend) noexcept {
  backend.q8_input_chunk4_storage.clear();

  size_t max_block_count = 0u;
  if (!update_q8_input_requirement(backend.output, max_block_count) ||
      !update_q8_input_requirement(backend.output_argmax, max_block_count)) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? update_q8_input_requirement(block.attention_q, max_block_count) &&
              update_q8_input_requirement(block.attention_k, max_block_count) &&
              update_q8_input_requirement(block.attention_v, max_block_count) &&
              update_q8_input_requirement(block.attention_output, max_block_count)
        : update_q8_input_requirement(block.shortconv_in_proj, max_block_count) &&
              update_q8_input_requirement(block.shortconv_out_proj, max_block_count);
    if (!residual_ok ||
        !update_q8_input_requirement(block.feed_forward_gate, max_block_count) ||
        !update_q8_input_requirement(block.feed_forward_down, max_block_count) ||
        !update_q8_input_requirement(block.feed_forward_up, max_block_count)) {
      return false;
    }
  }

  if (max_block_count == 0u) {
    return true;
  }

  backend.q8_input_chunk4_storage.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * max_block_count);
  return true;
}

inline bool prepare_q8_input_chunk8_workspace(native_backend & backend) noexcept {
  backend.q8_input_chunk8_storage.clear();

  size_t max_block_count = 0u;
  if (!update_q8_input_requirement(backend.output, max_block_count) ||
      !update_q8_input_requirement(backend.output_argmax, max_block_count)) {
    return false;
  }

  for (const auto & block : backend.blocks) {
    const bool residual_ok = block.uses_attention
        ? update_q8_input_requirement(block.attention_q, max_block_count) &&
              update_q8_input_requirement(block.attention_k, max_block_count) &&
              update_q8_input_requirement(block.attention_v, max_block_count) &&
              update_q8_input_requirement(block.attention_output, max_block_count)
        : update_q8_input_requirement(block.shortconv_in_proj, max_block_count) &&
              update_q8_input_requirement(block.shortconv_out_proj, max_block_count);
    if (!residual_ok ||
        !update_q8_input_requirement(block.feed_forward_gate, max_block_count) ||
        !update_q8_input_requirement(block.feed_forward_down, max_block_count) ||
        !update_q8_input_requirement(block.feed_forward_up, max_block_count)) {
      return false;
    }
  }

  if (max_block_count == 0u) {
    return true;
  }

  backend.q8_input_chunk8_storage.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * max_block_count);
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
    const bool residual_ok = block.uses_attention
        ? update_packed_q8_0_input_requirement(block.attention_q, max_block_count) &&
              update_packed_q8_0_input_requirement(block.attention_k, max_block_count) &&
              update_packed_q8_0_input_requirement(block.attention_v, max_block_count) &&
              update_packed_q8_0_input_requirement(block.attention_output, max_block_count)
        : update_packed_q8_0_input_requirement(block.shortconv_in_proj, max_block_count) &&
              update_packed_q8_0_input_requirement(block.shortconv_out_proj, max_block_count);
    if (!residual_ok ||
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
    const bool residual_ok = block.uses_attention
        ? update_packed_q8_0_input_requirement(block.attention_q, max_block_count) &&
              update_packed_q8_0_input_requirement(block.attention_k, max_block_count) &&
              update_packed_q8_0_input_requirement(block.attention_v, max_block_count) &&
              update_packed_q8_0_input_requirement(block.attention_output, max_block_count)
        : update_packed_q8_0_input_requirement(block.shortconv_in_proj, max_block_count) &&
              update_packed_q8_0_input_requirement(block.shortconv_out_proj, max_block_count);
    if (!residual_ok ||
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
      dtype == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
      dtype == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8 ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
      dtype == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
  const uint64_t storage_rows = dtype == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
          dtype == emel::kernel::detail::dtype_q8_0_x4_bl8
      ? emel::kernel::detail::quant::packed_q8_0_x4_group_count(rows)
      : (dtype == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
         dtype == emel::kernel::detail::dtype_q4_k_x8_bl8)
      ? emel::kernel::detail::quant::packed_q4_k_x8_group_count(rows)
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
  if (matrix.tensor == nullptr) {
    return 0u;
  }

  const uint8_t dtype = static_cast<uint8_t>(matrix.tensor->type);
  const uint64_t storage_rows = dtype == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
          dtype == emel::kernel::detail::dtype_q8_0_x4_bl8
      ? emel::kernel::detail::quant::packed_q8_0_x4_group_count(static_cast<uint64_t>(matrix.rows))
      : (dtype == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
         dtype == emel::kernel::detail::dtype_q4_k_x8_bl8)
      ? emel::kernel::detail::quant::packed_q4_k_x8_group_count(
            static_cast<uint64_t>(matrix.rows))
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
    if (block.uses_attention) {
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
    } else {
      append_leaf_lifecycle_tensor(
          backend,
          const_cast<void *>(block.shortconv_in_proj.tensor->data),
          matrix_buffer_bytes(block.shortconv_in_proj));
      append_leaf_lifecycle_tensor(
          backend,
          const_cast<void *>(block.shortconv_out_proj.tensor->data),
          matrix_buffer_bytes(block.shortconv_out_proj));
      append_leaf_lifecycle_tensor(
          backend,
          block.shortconv_conv.data(),
          static_cast<uint64_t>(block.shortconv_conv.size()) * sizeof(float));
    }
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

inline const emel::graph::processor::event::lifecycle_manifest * prefill_lifecycle(
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
  return &backend.prefill_lifecycle;
}

inline const emel::graph::processor::event::lifecycle_manifest * decode_lifecycle(
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
  return &backend.decode_lifecycle;
}

inline bool prepare_packed_q8_0_input(native_backend & backend,
                                      std::span<const float> input) noexcept;

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector_prepared_packed_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    const int32_t input_cols,
    std::span<float> output) noexcept;

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector_q8_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    std::span<const emel::kernel::detail::quant::block_q8_k> input,
    const int32_t input_cols,
    std::span<float> output) noexcept;

inline bool prepare_packed_q8_0_chunk4_input(native_backend & backend,
                                             std::span<const float> input,
                                             const int32_t input_cols) noexcept;

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_chunk4_prepared_packed_q8_0_input(
    native_backend & backend,
    const tensor_matrix & matrix,
    const int32_t input_cols,
    std::span<float> output) noexcept;

inline bool prepare_q8_chunk4_input(native_backend & backend,
                                    std::span<const float> input,
                                    const int32_t input_cols) noexcept;

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_chunk4_q8_input(native_backend & backend,
                                   const tensor_matrix & matrix,
                                   const int32_t input_cols,
                                   std::span<float> output) noexcept;

inline bool quantize_vector_q8_k(
    std::span<const float> input,
    std::span<emel::kernel::detail::quant::block_q8_k> output) noexcept;

inline bool valid_matmul_vector_shape(const tensor_matrix & matrix,
                                      std::span<const float> input,
                                      std::span<float> output) noexcept {
  return matrix.cols > 0 &&
      matrix.rows > 0 &&
      static_cast<size_t>(matrix.cols) == input.size() &&
      static_cast<size_t>(matrix.rows) == output.size();
}

struct matmul_row_slice {
  int32_t row_begin = 0;
  int32_t row_count = 0;
};

// Row-group granularity for view slicing: packed interleaved dtypes store
// multiple logical rows per storage group, so slice boundaries must land on
// group multiples; plain row-major dtypes slice on any row boundary.
inline uint64_t matmul_slice_group_rows(const emel::kernel::event::dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  const uint64_t x8_group =
      static_cast<uint64_t>(code == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
                            code == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
                            code == emel::kernel::detail::dtype_q6_k_x8 ||
                            code == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
                            code == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared) *
      emel::kernel::detail::quant::Q4_K_X8_ROWS;
  const uint64_t x4_group =
      static_cast<uint64_t>(code == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
                            code == emel::kernel::detail::dtype_q8_0_x4_bl8) *
      emel::kernel::detail::quant::Q8_0_X4_ROWS;
  return std::max<uint64_t>(x8_group + x4_group, 1u);
}

// Partition weight rows into at most k_matmul_lanes contiguous group-aligned
// slices. Pure bounded partition arithmetic; a ragged tail (rows not a group
// multiple) lands in the final slice, matching the padded storage group the
// packed formats already carry.
inline size_t compute_matmul_row_slices(
    const uint64_t rows,
    const uint64_t group_rows,
    std::array<matmul_row_slice, k_matmul_lanes> & slices) noexcept {
  const uint64_t groups = (rows + group_rows - 1u) / group_rows;
  const uint64_t lane_count = std::min<uint64_t>(k_matmul_lanes, std::max<uint64_t>(groups, 1u));
  const uint64_t groups_per_lane = groups / lane_count;
  const uint64_t extra_groups = groups % lane_count;
  uint64_t begin_group = 0u;
  for (uint64_t lane = 0u; lane < lane_count; ++lane) {
    const uint64_t lane_groups = groups_per_lane + static_cast<uint64_t>(lane < extra_groups);
    const uint64_t begin_row = begin_group * group_rows;
    const uint64_t end_row = std::min(rows, (begin_group + lane_groups) * group_rows);
    slices[lane].row_begin = static_cast<int32_t>(begin_row);
    slices[lane].row_count = static_cast<int32_t>(end_row - begin_row);
    begin_group += lane_groups;
  }
  return static_cast<size_t>(lane_count);
}

// Derive a slice event from a full mul_mat event: sliced src0/dst views over
// a contiguous group-aligned row range, shared src1 input. The event remains
// a complete work description; lane kernels never learn about slicing.
inline emel::kernel::event::op_mul_mat compute_sliced_mul_mat_event(
    const emel::kernel::event::op_mul_mat & ev,
    const uint64_t group_rows,
    const matmul_row_slice slice) noexcept {
  emel::kernel::event::op_mul_mat sliced = ev;
  const uint64_t begin = static_cast<uint64_t>(slice.row_begin);
  const uint64_t count = static_cast<uint64_t>(slice.row_count);
  const uint64_t slice_groups = (count + group_rows - 1u) / group_rows;
  sliced.src0.data =
      static_cast<const uint8_t *>(ev.src0.data) + (begin / group_rows) * ev.src0.nb[1];
  sliced.src0.ne[1] = count;
  sliced.src0.nb[2] = ev.src0.nb[1] * slice_groups;
  sliced.src0.nb[3] = sliced.src0.nb[2];
  sliced.dst.data = static_cast<uint8_t *>(ev.dst.data) + begin * ev.dst.nb[1];
  sliced.dst.ne[1] = count;
  sliced.dst.nb[2] = ev.dst.nb[1] * count;
  sliced.dst.nb[3] = sliced.dst.nb[2];
  return sliced;
}

// Fork/join slice dispatch across the lane kernel actors. The caller computes
// the first slice while pool workers compute the rest; every slice joins
// before this helper returns, so no work escapes the RTC boundary. Slices
// write disjoint dst rows and reorder no reductions, so the output is
// bit-identical to the serial dispatch.
inline bool compute_mul_mat_sliced_parallel(
    native_backend & backend,
    const emel::kernel::event::op_mul_mat & ev) noexcept {
  if (!backend.lane_pool.has_value() || ev.src0.ne[1] == 0u) {
    return false;
  }

  std::array<matmul_row_slice, k_matmul_lanes> slices = {};
  std::array<emel::kernel::event::op_mul_mat, k_matmul_lanes> lane_events = {};
  std::array<bool, k_matmul_lanes> lane_ok = {};
  const uint64_t group_rows = matmul_slice_group_rows(ev.src0.type);
  const size_t lane_count = compute_matmul_row_slices(ev.src0.ne[1], group_rows, slices);

  matmul_lane_scheduler scheduler{*backend.lane_pool};
  matmul_lane_scheduler::join_group group{};
  for (size_t lane = 1u; lane < lane_count; ++lane) {
    lane_events[lane] = compute_sliced_mul_mat_event(ev, group_rows, slices[lane]);
    auto & lane_kernel = backend.lane_kernels[lane];
    lane_kernel.set_kind(backend.kernel_kind);
    const auto & lane_ev = lane_events[lane];
    auto & ok_flag = lane_ok[lane];
    // Total fork: submit_or_run executes the slice exactly once inside the
    // join window (worker or calling thread - the scheduler's internal
    // capacity handling, never a behavior choice).
    scheduler.submit_or_run(group, [&lane_kernel, &lane_ev, &ok_flag]() noexcept {
      ok_flag = lane_kernel.process_event(lane_ev);
    });
  }
  lane_events[0] = compute_sliced_mul_mat_event(ev, group_rows, slices[0]);
  backend.lane_kernels[0].set_kind(backend.kernel_kind);
  lane_ok[0] = backend.lane_kernels[0].process_event(lane_events[0]);
  (void)group.wait();

  bool all_ok = true;
  for (size_t lane = 0u; lane < lane_count; ++lane) {
    all_ok = all_ok && lane_ok[lane];
  }
  return all_ok;
}

// Compile-time lane-mode seam for mul_mat dispatch. Route guards choose the
// lane mode; this helper only executes the already-chosen mode.
template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool compute_mul_mat(native_backend & backend,
                            const emel::kernel::event::op_mul_mat & ev) noexcept {
  if constexpr (lanes == matmul_lane_mode::parallel) {
    return compute_mul_mat_sliced_parallel(backend, ev);
  } else {
    backend.kernel.set_kind(backend.kernel_kind);
    return backend.kernel.process_event(ev);
  }
}

// Evidence counters live per kernel actor; parallel slices accrue on lane
// actors, so audit reads must sum the primary kernel and every lane.
template <class counter_fn>
inline uint64_t compute_kernel_counter_total(const native_backend & backend,
                                             counter_fn && counter) noexcept {
  uint64_t total = std::invoke(counter, backend.kernel);
  for (const auto & lane_kernel : backend.lane_kernels) {
    total += std::invoke(counter, lane_kernel);
  }
  return total;
}

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector_packed_q8_0(native_backend & backend,
                                      const tensor_matrix & matrix,
                                      std::span<const float> input,
                                      std::span<float> output) noexcept {
  if (!valid_matmul_vector_shape(matrix, input, output)) {
    return false;
  }

  return prepare_packed_q8_0_input(backend, input) &&
      matmul_vector_prepared_packed_q8_0_input<lanes>(backend, matrix, matrix.cols, output);
}

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector_q8_k(native_backend & backend,
                               const tensor_matrix & matrix,
                               std::span<const float> input,
                               std::span<float> output) noexcept {
  if (!valid_matmul_vector_shape(matrix, input, output)) {
    return false;
  }

  const size_t block_count = static_cast<size_t>(matrix.cols) / static_cast<size_t>(quant::QK_K);
  if (block_count == 0u || block_count > backend.q8_input_storage.size()) {
    return false;
  }

  auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
      backend.q8_input_storage.data(), block_count);
  return quantize_vector_q8_k(input, q8_input) &&
      matmul_vector_q8_input<lanes>(backend, matrix, q8_input, matrix.cols, output);
}

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector(native_backend & backend,
                          const tensor_matrix & matrix,
                          std::span<const float> input,
                          std::span<float> output) noexcept;

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector_native_quantized(native_backend & backend,
                                           const tensor_matrix & matrix,
                                           std::span<const float> input,
                                           std::span<float> output) noexcept {
  return matmul_vector<lanes>(backend, matrix, input, output);
}

template <matmul_lane_mode lanes>
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
          input.data(),
          emel::kernel::event::dtype::f32,
          static_cast<uint64_t>(1u),
          static_cast<uint64_t>(input.size())),
      .dst = make_dst_view(
          output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
  };
  const bool ok = compute_mul_mat<lanes>(backend, ev);
  backend.kernel_dispatch_calls += 1;
  backend.native_q8_0_dispatch_calls += static_cast<uint64_t>(
      matrix.tensor != nullptr &&
      static_cast<uint8_t>(matrix.tensor->type) == emel::kernel::detail::dtype_q8_0);
  return ok;
}

template <scalar_matmul_route route, matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_vector_routed(native_backend & backend,
                                 const tensor_matrix & matrix,
                                 std::span<const float> input,
                                 std::span<float> output) noexcept {
  if constexpr (route == scalar_matmul_route::packed_q8_0) {
    return matmul_vector_packed_q8_0<lanes>(backend, matrix, input, output);
  } else if constexpr (route == scalar_matmul_route::q8_k) {
    return matmul_vector_q8_k<lanes>(backend, matrix, input, output);
  } else if constexpr (route == scalar_matmul_route::native_quantized ||
                       route == scalar_matmul_route::native_quantized_q8_k_logits) {
    return matmul_vector_native_quantized<lanes>(backend, matrix, input, output);
  } else {
    return matmul_vector<lanes>(backend, matrix, input, output);
  }
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

template <matmul_lane_mode lanes>
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
  };
  const bool ok = compute_mul_mat<lanes>(backend, ev);
  backend.kernel_dispatch_calls += 1;
  return ok;
}

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
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
  };
  const bool ok = compute_mul_mat<lanes>(backend, ev);
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

inline bool prepare_q8_chunk4_input(native_backend & backend,
                                    std::span<const float> input,
                                    const int32_t input_cols) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(input_cols);
  if (backend.q8_input_chunk4_storage.empty() ||
      input_cols <= 0 ||
      input.size() != expected_size ||
      (static_cast<size_t>(input_cols) % quant::QK_K) != 0u) {
    return false;
  }

  const size_t block_count = static_cast<size_t>(input_cols) / quant::QK_K;
  const size_t required_blocks =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * block_count;
  if (required_blocks > backend.q8_input_chunk4_storage.size()) {
    return false;
  }

  for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
    if (!quantize_vector_q8_k(
            input.subspan(
                static_cast<size_t>(row) * static_cast<size_t>(input_cols),
                static_cast<size_t>(input_cols)),
            std::span<emel::kernel::detail::quant::block_q8_k>(
                backend.q8_input_chunk4_storage.data() +
                    static_cast<size_t>(row) * block_count,
                block_count))) {
      return false;
    }
  }

  return true;
}

inline bool prepare_q8_chunk8_input(native_backend & backend,
                                    std::span<const float> input,
                                    const int32_t input_cols) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(input_cols);
  if (backend.q8_input_chunk8_storage.empty() ||
      input_cols <= 0 ||
      input.size() != expected_size ||
      (static_cast<size_t>(input_cols) % quant::QK_K) != 0u) {
    return false;
  }

  const size_t block_count = static_cast<size_t>(input_cols) / quant::QK_K;
  const size_t required_blocks =
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * block_count;
  if (required_blocks > backend.q8_input_chunk8_storage.size()) {
    return false;
  }

  for (int32_t row = 0; row < k_prefill_q8_chunk8_rows; ++row) {
    if (!quantize_vector_q8_k(
            input.subspan(
                static_cast<size_t>(row) * static_cast<size_t>(input_cols),
                static_cast<size_t>(input_cols)),
            std::span<emel::kernel::detail::quant::block_q8_k>(
                backend.q8_input_chunk8_storage.data() +
                    static_cast<size_t>(row) * block_count,
                block_count))) {
      return false;
    }
  }

  return true;
}

template <matmul_lane_mode lanes>
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

  return matmul_vector_q8_0_input<lanes>(
      backend,
      matrix,
      std::span<const emel::kernel::detail::quant::block_q8_0>(
          backend.packed_q8_0_input_storage.data(), block_count),
      input_cols,
      output);
}

template <matmul_lane_mode lanes>
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
  };
  const bool ok = compute_mul_mat<lanes>(backend, ev);
  backend.kernel_dispatch_calls += 1;
  backend.packed_q8_0_dispatch_calls += static_cast<uint64_t>(ok);
  return ok;
}

template <matmul_lane_mode lanes>
inline bool matmul_chunk4_q8_input(native_backend & backend,
                                   const tensor_matrix & matrix,
                                   const int32_t input_cols,
                                   std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(matrix.rows);
  if (!q8_input_chunk4_path_supported(backend, matrix) ||
      input_cols <= 0 ||
      input_cols != matrix.cols ||
      output.size() != expected_size) {
    return false;
  }

  const size_t block_count = static_cast<size_t>(input_cols) / quant::QK_K;
  const size_t required_blocks =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * block_count;
  if (required_blocks > backend.q8_input_chunk4_storage.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_q8_k_rhs_chunk4_view(backend.q8_input_chunk4_storage.data(),
                                        static_cast<uint64_t>(input_cols)),
      .dst = make_batch_major_dst_view(
          output.data(),
          static_cast<uint64_t>(k_prefill_q8_chunk_rows),
          static_cast<uint64_t>(matrix.rows)),
  };
  const bool ok = compute_mul_mat<lanes>(backend, ev);
  backend.kernel_dispatch_calls += 1;
  return ok;
}

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_chunk8_q8_input(native_backend & backend,
                                   const tensor_matrix & matrix,
                                   const int32_t input_cols,
                                   std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(matrix.rows);
  if (!q8_input_chunk8_path_supported(backend, matrix) ||
      input_cols <= 0 ||
      input_cols != matrix.cols ||
      output.size() != expected_size) {
    return false;
  }

  const size_t block_count = static_cast<size_t>(input_cols) / quant::QK_K;
  const size_t required_blocks =
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * block_count;
  if (required_blocks > backend.q8_input_chunk8_storage.size()) {
    return false;
  }

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src_view(matrix),
      .src1 = make_q8_k_rhs_chunk8_view(backend.q8_input_chunk8_storage.data(),
                                        static_cast<uint64_t>(input_cols)),
      .dst = make_batch_major_dst_view(
          output.data(),
          static_cast<uint64_t>(k_prefill_q8_chunk8_rows),
          static_cast<uint64_t>(matrix.rows)),
  };
  const bool ok = compute_mul_mat<lanes>(backend, ev);
  backend.kernel_dispatch_calls += 1;
  return ok;
}

template <chunk4_rhs_route route>
inline bool prepare_chunk4_rhs(native_backend & backend,
                               std::span<const float> input,
                               const int32_t input_cols) noexcept {
  if constexpr (route == chunk4_rhs_route::packed_q8_0) {
    return prepare_packed_q8_0_chunk4_input(backend, input, input_cols);
  } else {
    return prepare_q8_chunk4_input(backend, input, input_cols);
  }
}

template <chunk4_rhs_route route, matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool matmul_chunk4_prepared(native_backend & backend,
                                   const tensor_matrix & matrix,
                                   const int32_t input_cols,
                                   std::span<float> output) noexcept {
  if constexpr (route == chunk4_rhs_route::packed_q8_0) {
    return matmul_chunk4_prepared_packed_q8_0_input<lanes>(backend, matrix, input_cols, output);
  } else {
    return matmul_chunk4_q8_input<lanes>(backend, matrix, input_cols, output);
  }
}

inline bool matmul_chunk4(native_backend & backend,
                          const tensor_matrix & matrix,
                          std::span<const float> input,
                          const int32_t input_cols,
                          std::span<float> output) noexcept {
  const size_t expected_input =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(input_cols);
  const size_t expected_output =
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(matrix.rows);
  if (input_cols <= 0 ||
      input.size() != expected_input ||
      output.size() != expected_output) {
    return false;
  }

  if (packed_q8_0_chunk4_input_path_supported(backend, matrix)) {
    return prepare_packed_q8_0_chunk4_input(backend, input, input_cols) &&
        matmul_chunk4_prepared_packed_q8_0_input(backend, matrix, input_cols, output);
  }

  if (q8_input_chunk4_path_supported(backend, matrix)) {
    return prepare_q8_chunk4_input(backend, input, input_cols) &&
        matmul_chunk4_q8_input(backend, matrix, input_cols, output);
  }

  return false;
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

template <int32_t chunk_rows, class value_type>
inline std::span<value_type> chunk_row_span(std::span<value_type> values,
                                            const int32_t row,
                                            const int32_t cols) noexcept {
  return values.subspan(
      static_cast<size_t>(row) * static_cast<size_t>(cols),
      static_cast<size_t>(cols));
}

template <int32_t chunk_rows, class value_type>
inline std::span<const value_type> chunk_row_span(std::span<const value_type> values,
                                                  const int32_t row,
                                                  const int32_t cols) noexcept {
  return values.subspan(
      static_cast<size_t>(row) * static_cast<size_t>(cols),
      static_cast<size_t>(cols));
}

template <class value_type>
inline std::span<value_type> chunk4_row_span(std::span<value_type> values,
                                             const int32_t row,
                                             const int32_t cols) noexcept {
  return chunk_row_span<k_prefill_q8_chunk_rows>(values, row, cols);
}

template <class value_type>
inline std::span<const value_type> chunk4_row_span(std::span<const value_type> values,
                                                   const int32_t row,
                                                   const int32_t cols) noexcept {
  return chunk_row_span<k_prefill_q8_chunk_rows>(values, row, cols);
}

template <class value_type>
inline std::span<value_type> chunk8_row_span(std::span<value_type> values,
                                             const int32_t row,
                                             const int32_t cols) noexcept {
  return chunk_row_span<k_prefill_q8_chunk8_rows>(values, row, cols);
}

template <class value_type>
inline std::span<const value_type> chunk8_row_span(std::span<const value_type> values,
                                                   const int32_t row,
                                                   const int32_t cols) noexcept {
  return chunk_row_span<k_prefill_q8_chunk8_rows>(values, row, cols);
}

template <int32_t chunk_rows>
inline bool rms_norm_chunked(std::span<const float> input,
                             const int32_t cols,
                             std::span<const float> weight,
                             const float epsilon,
                             std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(chunk_rows) * static_cast<size_t>(cols);
  if (cols <= 0 || input.size() != expected_size || output.size() != expected_size) {
    return false;
  }

  for (int32_t row = 0; row < chunk_rows; ++row) {
    if (!rms_norm(
            chunk_row_span<chunk_rows, const float>(input, row, cols),
            weight,
            epsilon,
            chunk_row_span<chunk_rows, float>(output, row, cols))) {
      return false;
    }
  }
  return true;
}

inline bool rms_norm_chunk4(std::span<const float> input,
                            const int32_t cols,
                            std::span<const float> weight,
                            const float epsilon,
                            std::span<float> output) noexcept {
  return rms_norm_chunked<k_prefill_q8_chunk_rows>(input, cols, weight, epsilon, output);
}

inline bool rms_norm_chunk8(std::span<const float> input,
                            const int32_t cols,
                            std::span<const float> weight,
                            const float epsilon,
                            std::span<float> output) noexcept {
  return rms_norm_chunked<k_prefill_q8_chunk8_rows>(input, cols, weight, epsilon, output);
}

template <int32_t chunk_rows>
inline bool add_chunk_rows_in_place(std::span<float> dst,
                                    std::span<const float> src,
                                    const int32_t cols) noexcept {
  const size_t expected_size =
      static_cast<size_t>(chunk_rows) * static_cast<size_t>(cols);
  if (cols <= 0 || dst.size() != expected_size || src.size() != expected_size) {
    return false;
  }

  for (size_t idx = 0; idx < expected_size; ++idx) {
    dst[idx] += src[idx];
  }
  return true;
}

inline bool add_chunk4_rows_in_place(std::span<float> dst,
                                     std::span<const float> src,
                                     const int32_t cols) noexcept {
  return add_chunk_rows_in_place<k_prefill_q8_chunk_rows>(dst, src, cols);
}

inline bool add_chunk8_rows_in_place(std::span<float> dst,
                                     std::span<const float> src,
                                     const int32_t cols) noexcept {
  return add_chunk_rows_in_place<k_prefill_q8_chunk8_rows>(dst, src, cols);
}

inline float silu(const float value) noexcept;

template <int32_t chunk_rows>
inline bool apply_silu_mul_chunked(std::span<const float> gate,
                                   std::span<const float> up,
                                   const int32_t cols,
                                   std::span<float> output) noexcept {
  const size_t expected_size =
      static_cast<size_t>(chunk_rows) * static_cast<size_t>(cols);
  if (cols <= 0 || gate.size() != expected_size || up.size() != expected_size ||
      output.size() != expected_size) {
    return false;
  }

  for (size_t idx = 0; idx < expected_size; ++idx) {
    output[idx] = silu(gate[idx]) * up[idx];
  }
  return true;
}

inline bool apply_silu_mul_chunk4(std::span<const float> gate,
                                  std::span<const float> up,
                                  const int32_t cols,
                                  std::span<float> output) noexcept {
  return apply_silu_mul_chunked<k_prefill_q8_chunk_rows>(gate, up, cols, output);
}

inline bool apply_silu_mul_chunk8(std::span<const float> gate,
                                  std::span<const float> up,
                                  const int32_t cols,
                                  std::span<float> output) noexcept {
  return apply_silu_mul_chunked<k_prefill_q8_chunk8_rows>(gate, up, cols, output);
}

inline void round_q_for_nonflash(std::span<const float> q_source,
                                 std::span<float> q_target) noexcept {
  for (size_t idx = 0; idx < q_source.size(); ++idx) {
    q_target[idx] = quant::fp16_to_fp32(quant::fp32_to_fp16(q_source[idx]));
  }
}

inline void apply_rope_pairing(std::span<float> vector,
                               const rope_pairing pairing,
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
  const int32_t pair_count = rot_dim / 2;
  const int32_t x1_base =
      pairing.x1_offset + pairing.x1_half_rot_offset * pair_count;
  for (int32_t head = 0; head < head_count; ++head) {
    float * head_ptr =
        vector.data() + (static_cast<size_t>(head) * static_cast<size_t>(head_dim));
    float theta = static_cast<float>(position);
    for (int32_t pair = 0; pair < pair_count; ++pair) {
      const float cos_theta = ::cosf(theta);
      const float sin_theta = ::sinf(theta);
      const int32_t dim0 = pair * pairing.x0_stride;
      const int32_t dim1 = pair * pairing.x1_stride + x1_base;
      const float x0 = head_ptr[dim0];
      const float x1 = head_ptr[dim1];
      head_ptr[dim0] = x0 * cos_theta - x1 * sin_theta;
      head_ptr[dim1] = x0 * sin_theta + x1 * cos_theta;
      theta *= theta_scale;
    }
  }
}

inline void apply_rope(std::span<float> vector,
                       const int32_t head_count,
                       const int32_t head_dim,
                       const int32_t n_rot,
                       const int32_t position,
                       const float rope_freq_base) noexcept {
  apply_rope_pairing(vector,
                     normal_rope_pairing(),
                     head_count,
                     head_dim,
                     n_rot,
                     position,
                     rope_freq_base);
}

inline void apply_attention_rope(std::span<float> vector,
                                 const block_weights & block,
                                 const int32_t head_count,
                                 const int32_t head_dim,
                                 const int32_t n_rot,
                                 const int32_t position,
                                 const float rope_freq_base) noexcept {
  apply_rope_pairing(vector,
                     block.attention_rope_pairing,
                     head_count,
                     head_dim,
                     n_rot,
                     position,
                     rope_freq_base);
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

// Logical->physical KV position through the dispatch-local block table.
inline size_t physical_kv_position(const kv_addressing_view & kv,
                                   const int32_t position) noexcept {
  const int32_t block_id = kv.blocks[static_cast<size_t>(position / kv.block_tokens)];
  return static_cast<size_t>(block_id * kv.block_tokens + (position % kv.block_tokens));
}

inline size_t layer_cache_offset(const native_backend & backend,
                                 const kv_addressing_view & kv,
                                 const block_weights & block,
                                 const int32_t layer,
                                 const int32_t position) noexcept {
  const size_t physical_position = physical_kv_position(kv, position);
  if (backend.layer_cache_offsets.size() != static_cast<size_t>(backend.n_layer)) {
    return ((static_cast<size_t>(layer) * static_cast<size_t>(backend.kv_positions_capacity)) +
            physical_position) *
        static_cast<size_t>(effective_attention_kv_dim(backend, block));
  }
  return backend.layer_cache_offsets[static_cast<size_t>(layer)] +
         (physical_position *
          static_cast<size_t>(effective_attention_kv_dim(backend, block)));
}

inline size_t layer_cache_offset(const native_backend & backend,
                                 const block_weights & block,
                                 const int32_t layer,
                                 const int32_t position) noexcept {
  return layer_cache_offset(backend, identity_kv_addressing(), block, layer, position);
}

inline size_t flash_layer_cache_layer_offset(const native_backend & backend,
                                             const int32_t layer) noexcept {
  if (backend.flash_layer_cache_offsets.size() != static_cast<size_t>(backend.n_layer)) {
    return static_cast<size_t>(layer) *
        static_cast<size_t>(backend.n_head_kv) *
        static_cast<size_t>(backend.kv_positions_capacity) *
        static_cast<size_t>(backend.head_dim_kv);
  }
  return backend.flash_layer_cache_offsets[static_cast<size_t>(layer)];
}

inline size_t flash_layer_cache_head_offset(const native_backend & backend,
                                            const block_weights & block,
                                            const int32_t layer,
                                            const int32_t kv_head) noexcept {
  return flash_layer_cache_layer_offset(backend, layer) +
      static_cast<size_t>(kv_head) *
      static_cast<size_t>(backend.kv_positions_capacity) *
      static_cast<size_t>(effective_attention_head_dim_kv(backend, block));
}

inline size_t flash_layer_cache_head_position_offset(const native_backend & backend,
                                                     const kv_addressing_view & kv,
                                                     const block_weights & block,
                                                     const int32_t layer,
                                                     const int32_t kv_head,
                                                     const int32_t position) noexcept {
  return flash_layer_cache_head_offset(backend, block, layer, kv_head) +
      physical_kv_position(kv, position) *
      static_cast<size_t>(effective_attention_head_dim_kv(backend, block));
}

inline size_t flash_layer_cache_head_position_offset(const native_backend & backend,
                                                     const block_weights & block,
                                                     const int32_t layer,
                                                     const int32_t kv_head,
                                                     const int32_t position) noexcept {
  return flash_layer_cache_head_position_offset(
      backend, identity_kv_addressing(), block, layer, kv_head, position);
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
  auto * io = static_cast<emel::text::generator::compute_io *>(request.compute_ctx);
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
                              const kv_addressing_view & kv,
                              const block_weights & block,
                              const int32_t layer_index,
                              const int32_t position_limit,
                              const std::span<const float> q_vector) noexcept {
  const int32_t head_count = backend.n_head;
  const int32_t head_dim = effective_attention_head_dim(backend, block);
  const int32_t kv_head_dim = effective_attention_head_dim_kv(backend, block);
  const int32_t q_dim = effective_attention_q_dim(backend, block);
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
          layer_cache_offset(backend, kv, block, layer_index, position) + kv_offset;
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
            layer_cache_offset(backend, kv, block, layer_index, position) + kv_offset;
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
    const block_weights & block,
    const int32_t layer_index,
    const int32_t position) noexcept {
  emel::kernel::event::op_flash_attn_ext request{};
  const uint64_t kv_tokens = static_cast<uint64_t>(position + 1);
  const uint64_t head_dim = static_cast<uint64_t>(effective_attention_head_dim(backend, block));
  const uint64_t head_count = static_cast<uint64_t>(backend.n_head);
  const uint64_t kv_head_dim =
      static_cast<uint64_t>(effective_attention_head_dim_kv(backend, block));
  const uint64_t kv_head_count = static_cast<uint64_t>(backend.n_head_kv);
  const size_t layer_offset = flash_layer_cache_layer_offset(backend, layer_index);
  const float scale =
      1.0f / std::sqrt(static_cast<float>(effective_attention_head_dim(backend, block)));
  const auto q_dim = static_cast<size_t>(effective_attention_q_dim(backend, block));
  const uint32_t masked_total_tokens = static_cast<uint32_t>(backend.n_ctx);
  const float * q_data = backend.q.size() >= q_dim ? backend.q.data()
                                                   : (backend.q_attn.size() >= q_dim
                                                          ? backend.q_attn.data()
                                                          : nullptr);
  auto attn_ctx =
      std::span<float>(const_cast<float *>(backend.attn_ctx.data()),
                       q_dim);

  request.src0 = make_src_view_3d(
      const_cast<float *>(q_data),
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
      sizeof(uint16_t) * static_cast<uint64_t>(backend.kv_positions_capacity) * kv_head_dim);
  request.src2 = make_src_view_strided_3d(
      const_cast<uint16_t *>(backend.flash_value_cache.data() + layer_offset),
      emel::kernel::event::dtype::f16,
      kv_head_dim,
      kv_tokens,
      kv_head_count,
      sizeof(uint16_t) * kv_head_dim,
      sizeof(uint16_t) * static_cast<uint64_t>(backend.kv_positions_capacity) * kv_head_dim);
  request.dst = make_dst_view_3d(
      attn_ctx.data(), head_dim, 1u, head_count);
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale),
              &masked_total_tokens,
              sizeof(masked_total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(masked_total_tokens);
  return request;
}

inline bool dispatch_flash_attention(native_backend & backend,
                                     const block_weights & block,
                                     const int32_t layer_index,
                                     const int32_t position) noexcept {
  const auto request = make_flash_attn_request(backend, block, layer_index, position);
  backend.kernel.set_kind(backend.kernel_kind);
  const bool ok = backend.kernel.process_event(request);
  ++backend.kernel_dispatch_calls;
  backend.flash_attention_dispatch_calls += static_cast<uint64_t>(ok);
  return ok;
}

inline bool store_attention_kv_cache(native_backend & backend,
                                     const kv_addressing_view & kv,
                                     const block_weights & block,
                                     const int32_t layer_index,
                                     const int32_t position,
                                     std::span<const float> k_vector,
                                     std::span<const float> v_vector) noexcept {
  const int32_t effective_kv_dim = effective_attention_kv_dim(backend, block);
  if (position < 0 ||
      position >= backend.n_ctx ||
      k_vector.size() != static_cast<size_t>(effective_kv_dim) ||
      v_vector.size() != static_cast<size_t>(effective_kv_dim)) {
    return false;
  }

  const size_t cache_offset = layer_cache_offset(backend, kv, block, layer_index, position);
  store_fp16_rounded_cache(k_vector, backend.key_cache.data() + cache_offset);
  store_fp16_rounded_cache(v_vector, backend.value_cache.data() + cache_offset);

  for (int32_t kv_head = 0; kv_head < backend.n_head_kv; ++kv_head) {
    const size_t src_offset =
        static_cast<size_t>(kv_head) *
        static_cast<size_t>(effective_attention_head_dim_kv(backend, block));
    const size_t flash_cache_offset = flash_layer_cache_head_position_offset(
        backend, kv, block, layer_index, kv_head, position);
    store_fp16_rounded_cache(
        k_vector.subspan(
            src_offset, static_cast<size_t>(effective_attention_head_dim_kv(backend, block))),
        backend.flash_key_cache.data() + flash_cache_offset);
    store_fp16_rounded_cache(
        v_vector.subspan(
            src_offset, static_cast<size_t>(effective_attention_head_dim_kv(backend, block))),
        backend.flash_value_cache.data() + flash_cache_offset);
  }

  return true;
}

template <emel::text::generator::attention_mode mode>
inline bool run_attention_for_q_vector(native_backend & backend,
                                       const kv_addressing_view & kv,
                                       const block_weights & block,
                                       const int32_t layer_index,
                                       const int32_t position,
                                       std::span<const float> q_vector) noexcept {
  if constexpr (mode == emel::text::generator::attention_mode::flash) {
    auto q = std::span<float>(backend.q.data(), q_vector.size());
    if (q_vector.size() != static_cast<size_t>(effective_attention_q_dim(backend, block))) {
      return false;
    }
    std::copy(q_vector.begin(), q_vector.end(), q.begin());
    return dispatch_flash_attention(backend, block, layer_index, position);
  } else {
    auto q_attn = std::span<float>(backend.q_attn.data(), q_vector.size());
    if (q_vector.size() != static_cast<size_t>(effective_attention_q_dim(backend, block))) {
      return false;
    }
    round_q_for_nonflash(q_vector, q_attn);
    return compute_attention(backend, kv, block, layer_index, position + 1, q_attn);
  }
}

template <emel::text::generator::attention_mode mode>
inline bool run_attention(native_backend & backend,
                          const kv_addressing_view & kv,
                          const block_weights & block,
                          const int32_t layer_index,
                          const int32_t position) noexcept {
  return run_attention_for_q_vector<mode>(
      backend,
      kv,
      block,
      layer_index,
      position,
      std::span<const float>(
          backend.q.data(), static_cast<size_t>(effective_attention_q_dim(backend, block))));
}

template <scalar_matmul_route route, matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_shortconv_block(native_backend & backend,
                                const kv_addressing_view & kv,
                                const block_weights & block,
                                const int32_t layer_index) noexcept {
  if (backend.shortconv_kernel_size <= 0 ||
      backend.shortconv_state_size <= 0 ||
      block.shortconv_in_proj.tensor == nullptr ||
      block.shortconv_out_proj.tensor == nullptr ||
      static_cast<size_t>(block.shortconv_in_proj.rows) !=
          static_cast<size_t>(3 * backend.n_embd) ||
      block.shortconv_in_proj.cols != backend.n_embd ||
      static_cast<size_t>(block.shortconv_out_proj.rows) !=
          static_cast<size_t>(backend.n_embd) ||
      block.shortconv_out_proj.cols != backend.n_embd ||
      block.shortconv_conv.size() !=
          static_cast<size_t>(backend.shortconv_kernel_size) *
              static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_bcx.size() != static_cast<size_t>(3 * backend.n_embd) ||
      backend.shortconv_bx.size() != static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_conv_out.size() != static_cast<size_t>(backend.n_embd)) {
    return false;
  }

  if (!matmul_vector_routed<route, lanes>(
          backend, block.shortconv_in_proj, backend.norm, backend.shortconv_bcx)) {
    return false;
  }

  auto b = std::span<const float>(
      backend.shortconv_bcx.data(), static_cast<size_t>(backend.n_embd));
  auto c = std::span<const float>(
      backend.shortconv_bcx.data() + static_cast<size_t>(backend.n_embd),
      static_cast<size_t>(backend.n_embd));
  auto x = std::span<const float>(
      backend.shortconv_bcx.data() + static_cast<size_t>(2 * backend.n_embd),
      static_cast<size_t>(backend.n_embd));

  const size_t layer_offset = shortconv_state_layer_offset(backend, kv, layer_index);
  float * state = backend.recurrent_shortconv_cache.data() + layer_offset;
  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    const size_t dim = static_cast<size_t>(idx);
    const float bx = b[dim] * x[dim];
    backend.shortconv_bx[dim] = bx;

    const float * kernel =
        block.shortconv_conv.data() + (dim * static_cast<size_t>(backend.shortconv_kernel_size));
    float conv_sum = bx * kernel[static_cast<size_t>(backend.shortconv_state_size)];
    for (int32_t tap = 0; tap < backend.shortconv_state_size; ++tap) {
      conv_sum += state[static_cast<size_t>(tap) * static_cast<size_t>(backend.n_embd) + dim] *
                  kernel[static_cast<size_t>(tap)];
    }

    backend.shortconv_conv_out[dim] = c[dim] * conv_sum;
  }

  if (backend.shortconv_state_size > 1) {
    const size_t move_count =
        static_cast<size_t>(backend.shortconv_state_size - 1) *
        static_cast<size_t>(backend.n_embd);
    std::memmove(
        state,
        state + static_cast<size_t>(backend.n_embd),
        move_count * sizeof(float));
  }
  std::memcpy(
      state + static_cast<size_t>(backend.shortconv_state_size - 1) *
                  static_cast<size_t>(backend.n_embd),
      backend.shortconv_bx.data(),
      static_cast<size_t>(backend.n_embd) * sizeof(float));

  if (!matmul_vector_routed<route, lanes>(
          backend, block.shortconv_out_proj, backend.shortconv_conv_out, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

template <scalar_matmul_route route, matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_shortconv_block(native_backend & backend,
                                const block_weights & block,
                                const int32_t layer_index) noexcept {
  return run_shortconv_block<route, lanes>(
      backend, identity_kv_addressing(), block, layer_index);
}

inline bool run_shortconv_block_chunk4(native_backend & backend,
                                       const kv_addressing_view & kv,
                                       const block_weights & block,
                                       const int32_t layer_index) noexcept {
  if (backend.shortconv_kernel_size <= 0 ||
      backend.shortconv_state_size <= 0 ||
      block.shortconv_in_proj.tensor == nullptr ||
      block.shortconv_out_proj.tensor == nullptr ||
      static_cast<size_t>(block.shortconv_in_proj.rows) !=
          static_cast<size_t>(3 * backend.n_embd) ||
      block.shortconv_in_proj.cols != backend.n_embd ||
      static_cast<size_t>(block.shortconv_out_proj.rows) !=
          static_cast<size_t>(backend.n_embd) ||
      block.shortconv_out_proj.cols != backend.n_embd ||
      block.shortconv_conv.size() !=
          static_cast<size_t>(backend.shortconv_kernel_size) *
              static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_bcx_chunk4.size() !=
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(3 * backend.n_embd) ||
      backend.shortconv_bx.size() != static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_conv_out_chunk4.size() != backend.hidden_chunk4.size()) {
    return false;
  }

  if (!matmul_chunk4(
          backend,
          block.shortconv_in_proj,
          backend.norm_chunk4,
          backend.n_embd,
          backend.shortconv_bcx_chunk4)) {
    return false;
  }

  const size_t layer_offset = shortconv_state_layer_offset(backend, kv, layer_index);
  float * state = backend.recurrent_shortconv_cache.data() + layer_offset;
  for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
    const auto bcx_row = chunk4_row_span<const float>(
        std::span<const float>(backend.shortconv_bcx_chunk4), row, 3 * backend.n_embd);
    auto conv_out_row = chunk4_row_span<float>(
        std::span<float>(backend.shortconv_conv_out_chunk4), row, backend.n_embd);
    auto b = bcx_row.subspan(0u, static_cast<size_t>(backend.n_embd));
    auto c = bcx_row.subspan(
        static_cast<size_t>(backend.n_embd), static_cast<size_t>(backend.n_embd));
    auto x = bcx_row.subspan(
        static_cast<size_t>(2 * backend.n_embd), static_cast<size_t>(backend.n_embd));

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      const size_t dim = static_cast<size_t>(idx);
      const float bx = b[dim] * x[dim];
      backend.shortconv_bx[dim] = bx;

      const float * kernel =
          block.shortconv_conv.data() + (dim * static_cast<size_t>(backend.shortconv_kernel_size));
      float conv_sum = bx * kernel[static_cast<size_t>(backend.shortconv_state_size)];
      for (int32_t tap = 0; tap < backend.shortconv_state_size; ++tap) {
        conv_sum += state[static_cast<size_t>(tap) * static_cast<size_t>(backend.n_embd) + dim] *
                    kernel[static_cast<size_t>(tap)];
      }

      conv_out_row[dim] = c[dim] * conv_sum;
    }

    if (backend.shortconv_state_size > 1) {
      const size_t move_count =
          static_cast<size_t>(backend.shortconv_state_size - 1) *
          static_cast<size_t>(backend.n_embd);
      std::memmove(
          state,
          state + static_cast<size_t>(backend.n_embd),
          move_count * sizeof(float));
    }
    std::memcpy(
        state + static_cast<size_t>(backend.shortconv_state_size - 1) *
                    static_cast<size_t>(backend.n_embd),
        backend.shortconv_bx.data(),
        static_cast<size_t>(backend.n_embd) * sizeof(float));
  }

  if (!matmul_chunk4(
          backend,
          block.shortconv_out_proj,
          backend.shortconv_conv_out_chunk4,
          backend.n_embd,
          backend.projected_chunk4) ||
      !add_chunk4_rows_in_place(backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd)) {
    return false;
  }

  return true;
}

inline bool run_shortconv_block_chunk4(native_backend & backend,
                                       const block_weights & block,
                                       const int32_t layer_index) noexcept {
  return run_shortconv_block_chunk4(backend, identity_kv_addressing(), block, layer_index);
}

// Records the prepare()-time matmul weight record pointers per layer in the
// canonical stream role order: pristine holds the bound records resident
// steps restore to (packed on aarch64), raw holds the untouched model
// records the streamed rebase clones (their dtype/extents describe the raw
// GGUF bytes the window copies into slots; a packed record would misread
// them). One-time engage bookkeeping (runs inside the initialize pipeline
// alongside prepare()'s own setup allocation).
inline void scan_stream_pristine_records(native_backend & backend) noexcept {
  backend.stream.pristine.assign(backend.blocks.size(), {});
  backend.stream.raw.assign(backend.blocks.size(), {});
  backend.stream.pristine_output = backend.output.tensor;
  backend.stream.raw_output = backend.output_native.tensor != nullptr
                                  ? backend.output_native.tensor
                                  : backend.output.tensor;
  backend.stream.pristine_output_argmax = backend.output_argmax.tensor;
  backend.stream.raw_output_argmax = backend.output_native.tensor != nullptr
                                         ? backend.output_native.tensor
                                         : backend.output_argmax.tensor;
  for (size_t layer = 0; layer < backend.blocks.size(); ++layer) {
    const block_weights & block = backend.blocks[layer];
    auto & roles = backend.stream.pristine[layer];
    auto & raw = backend.stream.raw[layer];
    const auto record = [&](const size_t role, const tensor_matrix & matrix,
                            const packed_matrix_binding & packed) noexcept {
      roles[role] = matrix.tensor;
      raw[role] = packed.source != nullptr ? packed.source : matrix.tensor;
    };
    record(k_stream_role_attention_q, block.attention_q,
           block.attention_q_packed);
    record(k_stream_role_attention_k, block.attention_k,
           block.attention_k_packed);
    record(k_stream_role_attention_v, block.attention_v,
           block.attention_v_packed);
    record(k_stream_role_attention_output, block.attention_output,
           block.attention_output_packed);
    record(k_stream_role_feed_forward_gate, block.feed_forward_gate,
           block.feed_forward_gate_packed);
    record(k_stream_role_feed_forward_down, block.feed_forward_down,
           block.feed_forward_down_packed);
    record(k_stream_role_feed_forward_up, block.feed_forward_up,
           block.feed_forward_up_packed);
    record(k_stream_role_shortconv_in_proj, block.shortconv_in_proj,
           block.shortconv_in_proj_packed);
    record(k_stream_role_shortconv_out_proj, block.shortconv_out_proj,
           block.shortconv_out_proj_packed);
  }
}

// Rebases the layer's matmul weight views onto the acquired window slot:
// clones the raw model record per present role (canonical order matching the
// extent builder) with data repointed into the slot. The slot holds raw GGUF
// bytes, so the clone's dtype/extents must describe them - the prepare()-time
// record can be a packed repack on aarch64 and would misread the slot. Pure
// pointer bookkeeping on the already-chosen streamed route; absent-role
// skipping is data-plane presence filtering, and a count mismatch propagates
// as data-plane failure.
inline bool bind_streamed_block_views(
    native_backend & backend,
    const int32_t layer_index,
    const uint8_t * slot_base,
    const emel::model::tensor::window::detail::layer_descriptor & layout) noexcept {
  block_weights & block = backend.blocks[static_cast<size_t>(layer_index)];
  stream_binding & stream = backend.stream;
  const auto & raw = stream.raw[static_cast<size_t>(layer_index)];
  uint32_t extent_index = 0u;
  const auto rebind = [&](tensor_matrix & matrix, const size_t role) noexcept -> bool {
    if (raw[role] == nullptr) {
      return true;
    }
    if (extent_index >= layout.weight_count) {
      return false;
    }
    stream.records[role] = *raw[role];
    stream.records[role].data = slot_base + layout.weights[extent_index].slot_offset;
    matrix.tensor = &stream.records[role];
    extent_index += 1u;
    return true;
  };
  return rebind(block.attention_q, k_stream_role_attention_q) &&
         rebind(block.attention_k, k_stream_role_attention_k) &&
         rebind(block.attention_v, k_stream_role_attention_v) &&
         rebind(block.attention_output, k_stream_role_attention_output) &&
         rebind(block.feed_forward_gate, k_stream_role_feed_forward_gate) &&
         rebind(block.feed_forward_down, k_stream_role_feed_forward_down) &&
         rebind(block.feed_forward_up, k_stream_role_feed_forward_up) &&
         rebind(block.shortconv_in_proj, k_stream_role_shortconv_in_proj) &&
         rebind(block.shortconv_out_proj, k_stream_role_shortconv_out_proj) &&
         extent_index == layout.weight_count;
}

// Points the logits/argmax output views at the raw model records for a
// streamed step: the streamed route is classified from raw records, so a
// packed resident output would be misread by it. Pure pointer bookkeeping on
// the already-chosen streamed route.
inline void bind_streamed_output_views(native_backend & backend) noexcept {
  if (backend.stream.raw_output != nullptr) {
    backend.output.tensor = backend.stream.raw_output;
  }
  if (backend.stream.raw_output_argmax != nullptr) {
    backend.output_argmax.tensor = backend.stream.raw_output_argmax;
  }
}

// Restores every block's matmul weight views (and the output views) to the
// prepare()-time records after a streamed step, so a later resident step
// reads the per-layer tensors instead of the shared per-role stream records
// (which hold the last acquired slot's clone). Pure pointer bookkeeping on
// the already-completed streamed route; used by both streamed decode
// drivers.
inline void reset_stream_block_views(native_backend & backend) noexcept {
  if (backend.stream.pristine_output != nullptr) {
    backend.output.tensor = backend.stream.pristine_output;
  }
  if (backend.stream.pristine_output_argmax != nullptr) {
    backend.output_argmax.tensor = backend.stream.pristine_output_argmax;
  }
  // pristine is either empty (scan never ran) or exactly blocks.size()
  // (assigned in one shot by scan_stream_pristine_records), so its size
  // bounds the block indexing.
  const size_t layer_count = backend.stream.pristine.size();
  for (size_t layer = 0; layer < layer_count; ++layer) {
    block_weights & block = backend.blocks[layer];
    const auto & roles = backend.stream.pristine[layer];
    const auto restore = [&](tensor_matrix & matrix, const size_t role) noexcept {
      if (roles[role] != nullptr) {
        matrix.tensor = roles[role];
      }
    };
    restore(block.attention_q, k_stream_role_attention_q);
    restore(block.attention_k, k_stream_role_attention_k);
    restore(block.attention_v, k_stream_role_attention_v);
    restore(block.attention_output, k_stream_role_attention_output);
    restore(block.feed_forward_gate, k_stream_role_feed_forward_gate);
    restore(block.feed_forward_down, k_stream_role_feed_forward_down);
    restore(block.feed_forward_up, k_stream_role_feed_forward_up);
    restore(block.shortconv_in_proj, k_stream_role_shortconv_in_proj);
    restore(block.shortconv_out_proj, k_stream_role_shortconv_out_proj);
  }
}

struct stream_acquire_capture {
  bool done = false;
  const uint8_t * slot_base = nullptr;
  const emel::model::tensor::window::detail::layer_descriptor * layout = nullptr;

  static void on_done(
      void * object,
      const emel::model::tensor::window::events::acquire_layer_window_done & ev) noexcept {
    auto * capture = static_cast<stream_acquire_capture *>(object);
    capture->done = true;
    capture->slot_base = ev.slot_base;
    capture->layout = &ev.layout;
  }

  static void on_error(
      void *,
      const emel::model::tensor::window::events::acquire_layer_window_error &) noexcept {}
};

// Streamed layer residency: acquires the layer's window slot (suspending on
// the in-flight load when needed) and rebases the block's weight views into
// it. Failure propagates through the layer loop's bool data-plane error path.
inline bool acquire_streamed_layer(native_backend & backend,
                                   const int32_t layer_index) noexcept {
  stream_acquire_capture capture{};
  emel::model::tensor::window::event::acquire_layer_window acquire{layer_index};
  acquire.on_done = {&capture, &stream_acquire_capture::on_done};
  acquire.on_error = {&capture, &stream_acquire_capture::on_error};
  if (!backend.stream.window->process_event(acquire) || !capture.done) {
    return false;
  }
  return bind_streamed_block_views(backend, layer_index, capture.slot_base,
                                   *capture.layout);
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_layer(native_backend &backend, const int32_t layer_index,
                      const kv_addressing_view & kv,
                      const int32_t position,
                      int32_t * err_out = nullptr) noexcept {
  if constexpr (wmode == window_mode::streamed) {
    // Streamed instantiations are reached only from the decode wrappers,
    // which receive the graph processor's always-valid err pointer
    // (kernel_step run_callback passes &callback_err unconditionally).
    if (!acquire_streamed_layer(backend, layer_index)) {
      *err_out = k_error_stream_acquire;
      return false;
    }
  }
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  if (!rms_norm(backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  if (block.uses_attention) {
    const int32_t q_dim = effective_attention_q_dim(backend, block);
    const int32_t kv_dim = effective_attention_kv_dim(backend, block);
    auto q = std::span<float>(backend.q.data(), static_cast<size_t>(q_dim));
    auto k = std::span<float>(backend.k.data(), static_cast<size_t>(kv_dim));
    auto v = std::span<float>(backend.v.data(), static_cast<size_t>(kv_dim));
    auto attn_ctx =
        std::span<const float>(backend.attn_ctx.data(), static_cast<size_t>(q_dim));
    if constexpr (route == scalar_matmul_route::packed_q8_0) {
      if (!prepare_packed_q8_0_input(backend, backend.norm) ||
          !matmul_vector_prepared_packed_q8_0_input<lanes>(
              backend, block.attention_q, block.attention_q.cols, q) ||
          !matmul_vector_prepared_packed_q8_0_input<lanes>(
              backend, block.attention_k, block.attention_k.cols, k) ||
          !matmul_vector_prepared_packed_q8_0_input<lanes>(
              backend, block.attention_v, block.attention_v.cols, v)) {
        return false;
      }
    } else if constexpr (route == scalar_matmul_route::q8_k) {
      const size_t block_count =
          static_cast<size_t>(backend.n_embd) / static_cast<size_t>(quant::QK_K);
      if (block_count == 0u || block_count > backend.q8_input_storage.size()) {
        return false;
      }
      auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
          backend.q8_input_storage.data(), block_count);
      if (!quantize_vector_q8_k(backend.norm, q8_input) ||
          !matmul_vector_q8_input<lanes>(
              backend, block.attention_q, q8_input, block.attention_q.cols, q) ||
          !matmul_vector_q8_input<lanes>(
              backend, block.attention_k, q8_input, block.attention_k.cols, k) ||
          !matmul_vector_q8_input<lanes>(
              backend, block.attention_v, q8_input, block.attention_v.cols, v)) {
        return false;
      }
    } else if constexpr (route == scalar_matmul_route::native_quantized ||
                         route == scalar_matmul_route::native_quantized_q8_k_logits) {
      if (!matmul_vector_native_quantized<lanes>(backend, block.attention_q, backend.norm, q) ||
          !matmul_vector_native_quantized<lanes>(backend, block.attention_k, backend.norm, k) ||
          !matmul_vector_native_quantized<lanes>(backend, block.attention_v, backend.norm, v)) {
        return false;
      }
    } else {
      if (!matmul_vector<lanes>(backend, block.attention_q, backend.norm, q) ||
          !matmul_vector<lanes>(backend, block.attention_k, backend.norm, k) ||
          !matmul_vector<lanes>(backend, block.attention_v, backend.norm, v)) {
        return false;
      }
    }

    if (requires_attention_qk_norm(backend, block) &&
        !apply_attention_qk_norm(backend, block)) {
      return false;
    }
    if (requires_attention_v_norm(backend, layer_index, block) &&
        !apply_rms_norm_in_place(v, backend.rms_epsilon)) {
      return false;
    }

    apply_attention_rope(q,
                         block,
                         backend.n_head,
                         effective_attention_head_dim(backend, block),
                         effective_attention_rope_dim(backend, block),
                         position,
                         effective_attention_rope_freq_base(backend, block));
    apply_attention_rope(k,
                         block,
                         backend.n_head_kv,
                         effective_attention_head_dim_kv(backend, block),
                         effective_attention_rope_dim(backend, block),
                         position,
                         effective_attention_rope_freq_base(backend, block));
    if (!store_attention_kv_cache(
            backend,
            kv,
            block,
            layer_index,
            position,
            k,
            v) ||
        !run_attention<mode>(backend, kv, block, layer_index, position) ||
        !matmul_vector_routed<route, lanes>(
            backend, block.attention_output, attn_ctx, backend.projected)) {
      return false;
    }

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
    }
  } else if (!run_shortconv_block<route, lanes>(backend, kv, block, layer_index)) {
    return false;
  }

  if (!rms_norm(backend.hidden, block.feed_forward_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  const int32_t ffn_dim = block.feed_forward_gate.rows;
  auto gate = std::span<float>(backend.gate.data(), static_cast<size_t>(ffn_dim));
  auto up = std::span<float>(backend.up.data(), static_cast<size_t>(ffn_dim));
  auto ffn_hidden = std::span<float>(backend.ffn_hidden.data(), static_cast<size_t>(ffn_dim));
  if constexpr (route == scalar_matmul_route::packed_q8_0) {
    if (!prepare_packed_q8_0_input(backend, backend.norm) ||
        !matmul_vector_prepared_packed_q8_0_input<lanes>(
            backend, block.feed_forward_gate, block.feed_forward_gate.cols, gate) ||
        !matmul_vector_prepared_packed_q8_0_input<lanes>(
            backend, block.feed_forward_up, block.feed_forward_up.cols, up)) {
      return false;
    }
  } else if constexpr (route == scalar_matmul_route::q8_k) {
    const size_t block_count =
        static_cast<size_t>(backend.n_embd) / static_cast<size_t>(quant::QK_K);
    if (block_count == 0u || block_count > backend.q8_input_storage.size()) {
      return false;
    }
    auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
        backend.q8_input_storage.data(), block_count);
    if (!quantize_vector_q8_k(backend.norm, q8_input) ||
        !matmul_vector_q8_input<lanes>(
            backend,
            block.feed_forward_gate,
            q8_input,
            block.feed_forward_gate.cols,
            gate) ||
        !matmul_vector_q8_input<lanes>(
            backend,
            block.feed_forward_up,
            q8_input,
            block.feed_forward_up.cols,
            up)) {
      return false;
    }
  } else if constexpr (route == scalar_matmul_route::native_quantized ||
                       route == scalar_matmul_route::native_quantized_q8_k_logits) {
    if (!matmul_vector_native_quantized<lanes>(backend, block.feed_forward_gate, backend.norm, gate) ||
        !matmul_vector_native_quantized<lanes>(backend, block.feed_forward_up, backend.norm, up)) {
      return false;
    }
  } else {
    if (!matmul_vector<lanes>(backend, block.feed_forward_gate, backend.norm, gate) ||
        !matmul_vector<lanes>(backend, block.feed_forward_up, backend.norm, up)) {
      return false;
    }
  }

  for (size_t idx = 0; idx < gate.size(); ++idx) {
    ffn_hidden[idx] = silu(gate[idx]) * up[idx];
  }

  if (!matmul_vector_routed<route, lanes>(
          backend, block.feed_forward_down, ffn_hidden, backend.projected)) {
    return false;
  }

  for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
    backend.hidden[static_cast<size_t>(idx)] += backend.projected[static_cast<size_t>(idx)];
  }

  return true;
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_layer(native_backend &backend,
                      const int32_t layer_index,
                      const int32_t position,
                      int32_t * err_out = nullptr) noexcept {
  return run_layer<mode, route, lanes, wmode>(
      backend, layer_index, identity_kv_addressing(), position, err_out);
}

inline bool run_layer_flash(native_backend & backend,
                            const int32_t layer_index,
                            const int32_t position) noexcept {
  return run_layer<emel::text::generator::attention_mode::flash, scalar_matmul_route::kernel>(
      backend, layer_index, identity_kv_addressing(), position);
}

inline bool run_layer_nonflash(native_backend & backend,
                               const int32_t layer_index,
                               const int32_t position) noexcept {
  return run_layer<emel::text::generator::attention_mode::nonflash, scalar_matmul_route::kernel>(
      backend, layer_index, identity_kv_addressing(), position);
}

template <scalar_matmul_route route, matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool compute_logits(native_backend & backend) noexcept {
  if (!rms_norm(backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  if constexpr (route == scalar_matmul_route::packed_q8_0) {
    return prepare_packed_q8_0_input(backend, backend.norm) &&
        matmul_vector_prepared_packed_q8_0_input<lanes>(
               backend, backend.output, backend.n_embd, backend.bound_logits);
  } else if constexpr (route == scalar_matmul_route::q8_k) {
    const size_t block_count =
        static_cast<size_t>(backend.n_embd) / static_cast<size_t>(quant::QK_K);
    auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
        backend.q8_input_storage.data(), block_count);
    return quantize_vector_q8_k(backend.norm, q8_input) &&
        matmul_vector_q8_input<lanes>(
               backend,
               backend.output,
               q8_input,
               backend.n_embd,
               backend.bound_logits);
  } else if constexpr (route == scalar_matmul_route::native_quantized) {
    return matmul_vector_native_quantized<lanes>(
        backend, backend.output, backend.norm, backend.bound_logits);
  } else if constexpr (route == scalar_matmul_route::native_quantized_q8_k_logits) {
    const size_t block_count =
        static_cast<size_t>(backend.n_embd) / static_cast<size_t>(quant::QK_K);
    auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
        backend.q8_input_storage.data(), block_count);
    return quantize_vector_q8_k(backend.norm, q8_input) &&
        matmul_vector_q8_input<lanes>(
               backend,
               backend.output,
               q8_input,
               backend.n_embd,
               backend.bound_logits);
  } else {
    return matmul_vector<lanes>(backend, backend.output, backend.norm, backend.bound_logits);
  }
}

template <scalar_argmax_route route>
inline bool compute_logits_preselected_argmax(native_backend & backend,
                                              int32_t & selected_index,
                                              float & selected_score) noexcept {
  if (!rms_norm(backend.hidden, backend.output_norm, backend.rms_epsilon, backend.norm)) {
    return false;
  }

  const tensor_matrix & output_matrix =
      backend.output_argmax.tensor != nullptr ? backend.output_argmax : backend.output;
  if constexpr (route == scalar_argmax_route::q8_k) {
    const size_t block_count =
        static_cast<size_t>(backend.n_embd) / static_cast<size_t>(quant::QK_K);
    auto q8_input = std::span<emel::kernel::detail::quant::block_q8_k>(
        backend.q8_input_storage.data(), block_count);
    return quantize_vector_q8_k(backend.norm, q8_input) &&
        matmul_vector_q8_input_argmax(
               backend,
               output_matrix,
               q8_input,
               backend.n_embd,
               selected_index,
               selected_score);
  } else {
    return matmul_vector_argmax(
        backend, output_matrix, backend.norm, selected_index, selected_score);
  }
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route>
inline bool run_prefill_scalar_tokens(native_backend & backend,
                                      const kv_addressing_view & kv,
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
      if (!run_layer<mode, route>(backend, layer, kv, position)) {
        return false;
      }
    }
    backend.kv_cache_tokens = position + 1;
  }

  return true;
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route>
inline bool run_prefill_scalar_tokens(native_backend & backend,
                                      const size_t token_begin,
                                      const size_t token_end) noexcept {
  return run_prefill_scalar_tokens<mode, route>(
      backend, identity_kv_addressing(), token_begin, token_end);
}

template <chunk4_rhs_route route, matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_shortconv_block_chunk4(native_backend & backend,
                                       const kv_addressing_view & kv,
                                       const block_weights & block,
                                       const int32_t layer_index) noexcept {
  if (backend.shortconv_kernel_size <= 0 ||
      backend.shortconv_state_size <= 0 ||
      block.shortconv_in_proj.tensor == nullptr ||
      block.shortconv_out_proj.tensor == nullptr ||
      static_cast<size_t>(block.shortconv_in_proj.rows) !=
          static_cast<size_t>(3 * backend.n_embd) ||
      block.shortconv_in_proj.cols != backend.n_embd ||
      static_cast<size_t>(block.shortconv_out_proj.rows) !=
          static_cast<size_t>(backend.n_embd) ||
      block.shortconv_out_proj.cols != backend.n_embd ||
      block.shortconv_conv.size() !=
          static_cast<size_t>(backend.shortconv_kernel_size) *
              static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_bcx_chunk4.size() !=
          static_cast<size_t>(k_prefill_q8_chunk_rows) *
              static_cast<size_t>(3 * backend.n_embd) ||
      backend.shortconv_bx.size() != static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_conv_out_chunk4.size() != backend.hidden_chunk4.size()) {
    return false;
  }

  if (!prepare_chunk4_rhs<route>(backend, backend.norm_chunk4, backend.n_embd) ||
      !matmul_chunk4_prepared<route, lanes>(
          backend, block.shortconv_in_proj, backend.n_embd, backend.shortconv_bcx_chunk4)) {
    return false;
  }

  const size_t layer_offset = shortconv_state_layer_offset(backend, kv, layer_index);
  float * state = backend.recurrent_shortconv_cache.data() + layer_offset;
  for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
    const auto bcx_row = chunk4_row_span<const float>(
        std::span<const float>(backend.shortconv_bcx_chunk4), row, 3 * backend.n_embd);
    auto conv_out_row = chunk4_row_span<float>(
        std::span<float>(backend.shortconv_conv_out_chunk4), row, backend.n_embd);
    auto b = bcx_row.subspan(0u, static_cast<size_t>(backend.n_embd));
    auto c = bcx_row.subspan(
        static_cast<size_t>(backend.n_embd), static_cast<size_t>(backend.n_embd));
    auto x = bcx_row.subspan(
        static_cast<size_t>(2 * backend.n_embd), static_cast<size_t>(backend.n_embd));

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      const size_t dim = static_cast<size_t>(idx);
      const float bx = b[dim] * x[dim];
      backend.shortconv_bx[dim] = bx;

      const float * kernel =
          block.shortconv_conv.data() + (dim * static_cast<size_t>(backend.shortconv_kernel_size));
      float conv_sum = bx * kernel[static_cast<size_t>(backend.shortconv_state_size)];
      for (int32_t tap = 0; tap < backend.shortconv_state_size; ++tap) {
        conv_sum += state[static_cast<size_t>(tap) * static_cast<size_t>(backend.n_embd) + dim] *
                    kernel[static_cast<size_t>(tap)];
      }

      conv_out_row[dim] = c[dim] * conv_sum;
    }

    if (backend.shortconv_state_size > 1) {
      const size_t move_count =
          static_cast<size_t>(backend.shortconv_state_size - 1) *
          static_cast<size_t>(backend.n_embd);
      std::memmove(
          state,
          state + static_cast<size_t>(backend.n_embd),
          move_count * sizeof(float));
    }
    std::memcpy(
        state + static_cast<size_t>(backend.shortconv_state_size - 1) *
                    static_cast<size_t>(backend.n_embd),
        backend.shortconv_bx.data(),
        static_cast<size_t>(backend.n_embd) * sizeof(float));
  }

  if (!prepare_chunk4_rhs<route>(backend, backend.shortconv_conv_out_chunk4, backend.n_embd) ||
      !matmul_chunk4_prepared<route, lanes>(
          backend, block.shortconv_out_proj, backend.n_embd, backend.projected_chunk4) ||
      !add_chunk4_rows_in_place(backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd)) {
    return false;
  }

  return true;
}

template <matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_shortconv_block_chunk8_q8_k(native_backend & backend,
                                            const kv_addressing_view & kv,
                                            const block_weights & block,
                                            const int32_t layer_index) noexcept {
  if (backend.shortconv_kernel_size <= 0 ||
      backend.shortconv_state_size <= 0 ||
      block.shortconv_in_proj.tensor == nullptr ||
      block.shortconv_out_proj.tensor == nullptr ||
      static_cast<size_t>(block.shortconv_in_proj.rows) !=
          static_cast<size_t>(3 * backend.n_embd) ||
      block.shortconv_in_proj.cols != backend.n_embd ||
      static_cast<size_t>(block.shortconv_out_proj.rows) !=
          static_cast<size_t>(backend.n_embd) ||
      block.shortconv_out_proj.cols != backend.n_embd ||
      block.shortconv_conv.size() !=
          static_cast<size_t>(backend.shortconv_kernel_size) *
              static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_bcx_chunk8.size() !=
          static_cast<size_t>(k_prefill_q8_chunk8_rows) *
              static_cast<size_t>(3 * backend.n_embd) ||
      backend.shortconv_bx.size() != static_cast<size_t>(backend.n_embd) ||
      backend.shortconv_conv_out_chunk8.size() != backend.hidden_chunk8.size()) {
    return false;
  }

  if (!prepare_q8_chunk8_input(backend, backend.norm_chunk8, backend.n_embd) ||
      !matmul_chunk8_q8_input<lanes>(
          backend, block.shortconv_in_proj, backend.n_embd, backend.shortconv_bcx_chunk8)) {
    return false;
  }

  const size_t layer_offset = shortconv_state_layer_offset(backend, kv, layer_index);
  float * state = backend.recurrent_shortconv_cache.data() + layer_offset;
  for (int32_t row = 0; row < k_prefill_q8_chunk8_rows; ++row) {
    const auto bcx_row = chunk8_row_span<const float>(
        std::span<const float>(backend.shortconv_bcx_chunk8), row, 3 * backend.n_embd);
    auto conv_out_row = chunk8_row_span<float>(
        std::span<float>(backend.shortconv_conv_out_chunk8), row, backend.n_embd);
    auto b = bcx_row.subspan(0u, static_cast<size_t>(backend.n_embd));
    auto c = bcx_row.subspan(
        static_cast<size_t>(backend.n_embd), static_cast<size_t>(backend.n_embd));
    auto x = bcx_row.subspan(
        static_cast<size_t>(2 * backend.n_embd), static_cast<size_t>(backend.n_embd));

    for (int32_t idx = 0; idx < backend.n_embd; ++idx) {
      const size_t dim = static_cast<size_t>(idx);
      const float bx = b[dim] * x[dim];
      backend.shortconv_bx[dim] = bx;

      const float * kernel =
          block.shortconv_conv.data() + (dim * static_cast<size_t>(backend.shortconv_kernel_size));
      float conv_sum = bx * kernel[static_cast<size_t>(backend.shortconv_state_size)];
      for (int32_t tap = 0; tap < backend.shortconv_state_size; ++tap) {
        conv_sum += state[static_cast<size_t>(tap) * static_cast<size_t>(backend.n_embd) + dim] *
                    kernel[static_cast<size_t>(tap)];
      }

      conv_out_row[dim] = c[dim] * conv_sum;
    }

    if (backend.shortconv_state_size > 1) {
      const size_t move_count =
          static_cast<size_t>(backend.shortconv_state_size - 1) *
          static_cast<size_t>(backend.n_embd);
      std::memmove(
          state,
          state + static_cast<size_t>(backend.n_embd),
          move_count * sizeof(float));
    }
    std::memcpy(
        state + static_cast<size_t>(backend.shortconv_state_size - 1) *
                    static_cast<size_t>(backend.n_embd),
        backend.shortconv_bx.data(),
        static_cast<size_t>(backend.n_embd) * sizeof(float));
  }

  if (!prepare_q8_chunk8_input(backend, backend.shortconv_conv_out_chunk8, backend.n_embd) ||
      !matmul_chunk8_q8_input<lanes>(
          backend, block.shortconv_out_proj, backend.n_embd, backend.projected_chunk8) ||
      !add_chunk8_rows_in_place(backend.hidden_chunk8, backend.projected_chunk8, backend.n_embd)) {
    return false;
  }

  return true;
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_layer_chunk4(native_backend & backend,
                             const kv_addressing_view & kv,
                             const int32_t layer_index,
                             const size_t token_base) noexcept {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  const int32_t q_dim = effective_attention_q_dim(backend, block);
  const int32_t kv_dim = effective_attention_kv_dim(backend, block);
  const int32_t ffn_dim = block.feed_forward_gate.rows;
  auto q_chunk =
      std::span<float>(backend.q_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(q_dim));
  auto k_chunk =
      std::span<float>(backend.k_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(kv_dim));
  auto v_chunk =
      std::span<float>(backend.v_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(kv_dim));
  auto attn_ctx_chunk =
      std::span<float>(backend.attn_ctx_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(q_dim));

  if (!rms_norm_chunk4(
          backend.hidden_chunk4,
          backend.n_embd,
          block.attention_norm,
          backend.rms_epsilon,
          backend.norm_chunk4)) {
    return false;
  }

  if (block.uses_attention) {
    if (!prepare_chunk4_rhs<route>(backend, backend.norm_chunk4, backend.n_embd) ||
        !matmul_chunk4_prepared<route, lanes>(
            backend, block.attention_q, backend.n_embd, q_chunk) ||
        !matmul_chunk4_prepared<route, lanes>(
            backend, block.attention_k, backend.n_embd, k_chunk) ||
        !matmul_chunk4_prepared<route, lanes>(
            backend, block.attention_v, backend.n_embd, v_chunk)) {
      return false;
    }

    const bool qk_norm_runtime = requires_attention_qk_norm(backend, block);
    const bool v_norm_runtime = requires_attention_v_norm(backend, layer_index, block);
    for (int32_t row = 0; row < k_prefill_q8_chunk_rows; ++row) {
      const int32_t position = backend.bound_positions[token_base + static_cast<size_t>(row)];
      auto q_row = chunk4_row_span<float>(q_chunk, row, q_dim);
      auto k_row = chunk4_row_span<float>(k_chunk, row, kv_dim);
      auto v_row = chunk4_row_span<float>(v_chunk, row, kv_dim);

      if (qk_norm_runtime &&
          (!apply_headwise_rms_norm(
               q_row,
               block.attention_q_norm,
               backend.n_head,
               effective_attention_head_dim(backend, block),
               backend.rms_epsilon) ||
           !apply_headwise_rms_norm(
               k_row,
               block.attention_k_norm,
               backend.n_head_kv,
               effective_attention_head_dim_kv(backend, block),
               backend.rms_epsilon))) {
        return false;
      }
      if (v_norm_runtime &&
          !apply_rms_norm_in_place(v_row, backend.rms_epsilon)) {
        return false;
      }

      apply_attention_rope(q_row,
                           block,
                           backend.n_head,
                           effective_attention_head_dim(backend, block),
                           effective_attention_rope_dim(backend, block),
                           position,
                           effective_attention_rope_freq_base(backend, block));
      apply_attention_rope(k_row,
                           block,
                           backend.n_head_kv,
                           effective_attention_head_dim_kv(backend, block),
                           effective_attention_rope_dim(backend, block),
                           position,
                           effective_attention_rope_freq_base(backend, block));

      if (!store_attention_kv_cache(backend, kv, block, layer_index, position, k_row, v_row) ||
          !run_attention_for_q_vector<mode>(backend, kv, block, layer_index, position, q_row)) {
        return false;
      }

      std::copy(
          backend.attn_ctx.begin(),
          backend.attn_ctx.begin() + q_dim,
          chunk4_row_span<float>(attn_ctx_chunk, row, q_dim).begin());
      backend.kv_cache_tokens = position + 1;
    }

    if (!prepare_chunk4_rhs<route>(backend, attn_ctx_chunk, q_dim) ||
        !matmul_chunk4_prepared<route, lanes>(
            backend, block.attention_output, q_dim, backend.projected_chunk4) ||
        !add_chunk4_rows_in_place(
            backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd)) {
      return false;
    }
  } else if (!run_shortconv_block_chunk4<route, lanes>(backend, kv, block, layer_index)) {
    return false;
  }

  if (!rms_norm_chunk4(
          backend.hidden_chunk4,
          backend.n_embd,
          block.feed_forward_norm,
          backend.rms_epsilon,
          backend.norm_chunk4)) {
    return false;
  }

  auto gate_chunk =
      std::span<float>(backend.gate_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(ffn_dim));
  auto up_chunk =
      std::span<float>(backend.up_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(ffn_dim));
  auto ffn_hidden_chunk =
      std::span<float>(backend.ffn_hidden_chunk4.data(),
                       static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(ffn_dim));
  if (!prepare_chunk4_rhs<route>(backend, backend.norm_chunk4, backend.n_embd) ||
      !matmul_chunk4_prepared<route, lanes>(
          backend, block.feed_forward_gate, backend.n_embd, gate_chunk) ||
      !matmul_chunk4_prepared<route, lanes>(
          backend, block.feed_forward_up, backend.n_embd, up_chunk)) {
    return false;
  }

  if (!apply_silu_mul_chunk4(
          gate_chunk, up_chunk, ffn_dim, ffn_hidden_chunk) ||
      !prepare_chunk4_rhs<route>(backend, ffn_hidden_chunk, ffn_dim) ||
      !matmul_chunk4_prepared<route, lanes>(
          backend, block.feed_forward_down, ffn_dim, backend.projected_chunk4) ||
      !add_chunk4_rows_in_place(backend.hidden_chunk4, backend.projected_chunk4, backend.n_embd)) {
    return false;
  }

  return true;
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_layer_chunk8_q8_k(native_backend & backend,
                                  const kv_addressing_view & kv,
                                  const int32_t layer_index,
                                  const size_t token_base) noexcept {
  auto & block = backend.blocks[static_cast<size_t>(layer_index)];
  const int32_t q_dim = effective_attention_q_dim(backend, block);
  const int32_t kv_dim = effective_attention_kv_dim(backend, block);
  const int32_t ffn_dim = block.feed_forward_gate.rows;
  auto q_chunk =
      std::span<float>(backend.q_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(q_dim));
  auto k_chunk =
      std::span<float>(backend.k_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(kv_dim));
  auto v_chunk =
      std::span<float>(backend.v_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(kv_dim));
  auto attn_ctx_chunk =
      std::span<float>(backend.attn_ctx_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(q_dim));

  if (!rms_norm_chunk8(
          backend.hidden_chunk8,
          backend.n_embd,
          block.attention_norm,
          backend.rms_epsilon,
          backend.norm_chunk8)) {
    return false;
  }

  if (block.uses_attention) {
    if (!prepare_q8_chunk8_input(backend, backend.norm_chunk8, backend.n_embd) ||
        !matmul_chunk8_q8_input<lanes>(backend, block.attention_q, backend.n_embd, q_chunk) ||
        !matmul_chunk8_q8_input<lanes>(backend, block.attention_k, backend.n_embd, k_chunk) ||
        !matmul_chunk8_q8_input<lanes>(backend, block.attention_v, backend.n_embd, v_chunk)) {
      return false;
    }

    const bool qk_norm_runtime = requires_attention_qk_norm(backend, block);
    const bool v_norm_runtime = requires_attention_v_norm(backend, layer_index, block);
    for (int32_t row = 0; row < k_prefill_q8_chunk8_rows; ++row) {
      const int32_t position = backend.bound_positions[token_base + static_cast<size_t>(row)];
      auto q_row = chunk8_row_span<float>(q_chunk, row, q_dim);
      auto k_row = chunk8_row_span<float>(k_chunk, row, kv_dim);
      auto v_row = chunk8_row_span<float>(v_chunk, row, kv_dim);

      if (qk_norm_runtime &&
          (!apply_headwise_rms_norm(
               q_row,
               block.attention_q_norm,
               backend.n_head,
               effective_attention_head_dim(backend, block),
               backend.rms_epsilon) ||
           !apply_headwise_rms_norm(
               k_row,
               block.attention_k_norm,
               backend.n_head_kv,
               effective_attention_head_dim_kv(backend, block),
               backend.rms_epsilon))) {
        return false;
      }
      if (v_norm_runtime &&
          !apply_rms_norm_in_place(v_row, backend.rms_epsilon)) {
        return false;
      }

      apply_attention_rope(q_row,
                           block,
                           backend.n_head,
                           effective_attention_head_dim(backend, block),
                           effective_attention_rope_dim(backend, block),
                           position,
                           effective_attention_rope_freq_base(backend, block));
      apply_attention_rope(k_row,
                           block,
                           backend.n_head_kv,
                           effective_attention_head_dim_kv(backend, block),
                           effective_attention_rope_dim(backend, block),
                           position,
                           effective_attention_rope_freq_base(backend, block));

      if (!store_attention_kv_cache(backend, kv, block, layer_index, position, k_row, v_row) ||
          !run_attention_for_q_vector<mode>(backend, kv, block, layer_index, position, q_row)) {
        return false;
      }

      std::copy(
          backend.attn_ctx.begin(),
          backend.attn_ctx.begin() + q_dim,
          chunk8_row_span<float>(attn_ctx_chunk, row, q_dim).begin());
      backend.kv_cache_tokens = position + 1;
    }

    if (!prepare_q8_chunk8_input(backend, attn_ctx_chunk, q_dim) ||
        !matmul_chunk8_q8_input<lanes>(
            backend, block.attention_output, q_dim, backend.projected_chunk8) ||
        !add_chunk8_rows_in_place(backend.hidden_chunk8, backend.projected_chunk8, backend.n_embd)) {
      return false;
    }
  } else if (!run_shortconv_block_chunk8_q8_k<lanes>(backend, kv, block, layer_index)) {
    return false;
  }

  if (!rms_norm_chunk8(
          backend.hidden_chunk8,
          backend.n_embd,
          block.feed_forward_norm,
          backend.rms_epsilon,
          backend.norm_chunk8)) {
    return false;
  }

  auto gate_chunk =
      std::span<float>(backend.gate_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(ffn_dim));
  auto up_chunk =
      std::span<float>(backend.up_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(ffn_dim));
  auto ffn_hidden_chunk =
      std::span<float>(backend.ffn_hidden_chunk8.data(),
                       static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(ffn_dim));
  if (!prepare_q8_chunk8_input(backend, backend.norm_chunk8, backend.n_embd) ||
      !matmul_chunk8_q8_input(
          backend, block.feed_forward_gate, backend.n_embd, gate_chunk) ||
      !matmul_chunk8_q8_input(
          backend, block.feed_forward_up, backend.n_embd, up_chunk)) {
    return false;
  }

  if (!apply_silu_mul_chunk8(
          gate_chunk, up_chunk, ffn_dim, ffn_hidden_chunk) ||
      !prepare_q8_chunk8_input(backend, ffn_hidden_chunk, ffn_dim) ||
      !matmul_chunk8_q8_input(
          backend, block.feed_forward_down, ffn_dim, backend.projected_chunk8) ||
      !add_chunk8_rows_in_place(backend.hidden_chunk8, backend.projected_chunk8, backend.n_embd)) {
    return false;
  }

  return true;
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk4_tokens(native_backend & backend,
                                      const kv_addressing_view & kv,
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
      if (!run_layer_chunk4<mode, route, lanes>(backend, kv, layer, token_base)) {
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

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk8_tokens_q8_k(native_backend & backend,
                                           const kv_addressing_view & kv,
                                           const size_t token_limit) noexcept {
  for (size_t token_base = 0; token_base < token_limit;
       token_base += static_cast<size_t>(k_prefill_q8_chunk8_rows)) {
    for (int32_t row = 0; row < k_prefill_q8_chunk8_rows; ++row) {
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
              chunk8_row_span<float>(
                  std::span<float>(backend.hidden_chunk8), row, backend.n_embd))) {
        return false;
      }
    }

    for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
      if (!run_layer_chunk8_q8_k<mode, lanes>(backend, kv, layer, token_base)) {
        return false;
      }
    }

    std::copy(
        chunk8_row_span<const float>(
            std::span<const float>(backend.hidden_chunk8),
            k_prefill_q8_chunk8_rows - 1,
            backend.n_embd)
            .begin(),
        chunk8_row_span<const float>(
            std::span<const float>(backend.hidden_chunk8),
            k_prefill_q8_chunk8_rows - 1,
            backend.n_embd)
            .end(),
        backend.hidden.begin());
  }

  return true;
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route>
inline bool run_prefill(native_backend & backend, const kv_addressing_view & kv) noexcept {
  backend.kv_cache_tokens = 0;
  reset_shortconv_cache(backend);

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  if (!run_prefill_scalar_tokens<mode, route>(backend, kv, 0u, token_count)) {
    return false;
  }

  return compute_logits<route>(backend);
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route>
inline bool run_prefill(native_backend & backend) noexcept {
  return run_prefill<mode, route>(backend, identity_kv_addressing());
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk4(native_backend & backend, const kv_addressing_view & kv) noexcept {
  backend.kv_cache_tokens = 0;
  reset_shortconv_cache(backend);

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_limit =
      token_count - (token_count % static_cast<size_t>(k_prefill_q8_chunk_rows));
  if (chunk_limit == 0u ||
      !run_prefill_chunk4_tokens<mode, route, lanes>(backend, kv, chunk_limit) ||
      !run_prefill_scalar_tokens<
          mode,
          static_cast<scalar_matmul_route>(route)>(backend, kv, chunk_limit, token_count)) {
    return false;
  }

  return compute_logits<static_cast<scalar_matmul_route>(route), lanes>(backend);
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk4(native_backend & backend) noexcept {
  return run_prefill_chunk4<mode, route, lanes>(backend, identity_kv_addressing());
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk8_q8_k(native_backend & backend,
                                    const kv_addressing_view & kv) noexcept {
  backend.kv_cache_tokens = 0;
  reset_shortconv_cache(backend);

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_limit =
      token_count - (token_count % static_cast<size_t>(k_prefill_q8_chunk8_rows));
  if (chunk_limit == 0u ||
      !run_prefill_chunk8_tokens_q8_k<mode, lanes>(backend, kv, chunk_limit) ||
      !run_prefill_scalar_tokens<mode, scalar_matmul_route::q8_k>(
          backend, kv, chunk_limit, token_count)) {
    return false;
  }

  return compute_logits<scalar_matmul_route::q8_k, lanes>(backend);
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk8_q8_k(native_backend & backend) noexcept {
  return run_prefill_chunk8_q8_k<mode, lanes>(backend, identity_kv_addressing());
}

inline bool run_prefill_flash(native_backend & backend) noexcept {
  return run_prefill<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel>(backend, identity_kv_addressing());
}

inline bool run_prefill_nonflash(native_backend & backend) noexcept {
  return run_prefill<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel>(backend, identity_kv_addressing());
}

template <emel::text::generator::attention_mode mode,
          scalar_matmul_route route,
          scalar_argmax_route argmax_route>
inline bool run_prefill_preselected_argmax(native_backend & backend,
                                           const kv_addressing_view & kv,
                                           int32_t & selected_index,
                                           float & selected_score) noexcept {
  backend.kv_cache_tokens = 0;
  reset_shortconv_cache(backend);
  if (!run_prefill_scalar_tokens<mode, route>(
          backend, kv, 0u, static_cast<size_t>(backend.bound_token_count))) {
    return false;
  }

  return compute_logits_preselected_argmax<argmax_route>(
      backend, selected_index, selected_score);
}

template <emel::text::generator::attention_mode mode,
          scalar_matmul_route route,
          scalar_argmax_route argmax_route>
inline bool run_prefill_preselected_argmax(native_backend & backend,
                                           int32_t & selected_index,
                                           float & selected_score) noexcept {
  return run_prefill_preselected_argmax<mode, route, argmax_route>(
      backend, identity_kv_addressing(), selected_index, selected_score);
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk4_preselected_argmax(native_backend & backend,
                                                  const kv_addressing_view & kv,
                                                  int32_t & selected_index,
                                                  float & selected_score) noexcept {
  backend.kv_cache_tokens = 0;
  reset_shortconv_cache(backend);

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_limit =
      token_count - (token_count % static_cast<size_t>(k_prefill_q8_chunk_rows));
  if (chunk_limit == 0u ||
      !run_prefill_chunk4_tokens<mode, route, lanes>(backend, kv, chunk_limit) ||
      !run_prefill_scalar_tokens<
          mode,
          static_cast<scalar_matmul_route>(route)>(backend, kv, chunk_limit, token_count)) {
    return false;
  }

  if constexpr (route == chunk4_rhs_route::q8_k) {
    return compute_logits_preselected_argmax<scalar_argmax_route::q8_k>(
        backend, selected_index, selected_score);
  } else {
    return compute_logits_preselected_argmax<scalar_argmax_route::kernel>(
        backend, selected_index, selected_score);
  }
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk4_preselected_argmax(native_backend & backend,
                                                  int32_t & selected_index,
                                                  float & selected_score) noexcept {
  return run_prefill_chunk4_preselected_argmax<mode, route, lanes>(
      backend, identity_kv_addressing(), selected_index, selected_score);
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk8_preselected_argmax_q8_k(native_backend & backend,
                                                       const kv_addressing_view & kv,
                                                       int32_t & selected_index,
                                                       float & selected_score) noexcept {
  backend.kv_cache_tokens = 0;
  reset_shortconv_cache(backend);

  const size_t token_count = static_cast<size_t>(backend.bound_token_count);
  const size_t chunk_limit =
      token_count - (token_count % static_cast<size_t>(k_prefill_q8_chunk8_rows));
  if (chunk_limit == 0u ||
      !run_prefill_chunk8_tokens_q8_k<mode, lanes>(backend, kv, chunk_limit) ||
      !run_prefill_scalar_tokens<mode, scalar_matmul_route::q8_k>(
          backend, kv, chunk_limit, token_count)) {
    return false;
  }

  return compute_logits_preselected_argmax<scalar_argmax_route::q8_k>(
      backend, selected_index, selected_score);
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_prefill_chunk8_preselected_argmax_q8_k(native_backend & backend,
                                                       int32_t & selected_index,
                                                       float & selected_score) noexcept {
  return run_prefill_chunk8_preselected_argmax_q8_k<mode, lanes>(
      backend, identity_kv_addressing(), selected_index, selected_score);
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_decode(native_backend &backend,
                       const emel::graph::processor::event::execute &request,
                       const kv_addressing_view & kv,
                       int32_t * err_out = nullptr) noexcept {
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

  if constexpr (wmode == window_mode::streamed) {
    bind_streamed_output_views(backend);
  }
  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!run_layer<mode, route, lanes, wmode>(backend, layer, kv, position,
                                              err_out)) {
      if constexpr (wmode == window_mode::streamed) {
        reset_stream_block_views(backend);
      }
      return false;
    }
  }
  backend.kv_cache_tokens = position + 1;
  if constexpr (wmode == window_mode::streamed) {
    // Logits run on the raw output view the streamed route was classified
    // for; the resident views are restored after.
    const bool logits_ok = compute_logits<route, lanes>(backend);
    reset_stream_block_views(backend);
    return logits_ok;
  } else {
    return compute_logits<route, lanes>(backend);
  }
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_decode(native_backend &backend,
                       const emel::graph::processor::event::execute &request,
                       int32_t * err_out = nullptr) noexcept {
  return run_decode<mode, route, lanes, wmode>(
      backend, request, identity_kv_addressing(), err_out);
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          scalar_argmax_route argmax_route,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_decode_preselected_argmax(
    native_backend &backend,
    const emel::graph::processor::event::execute &request,
    const kv_addressing_view & kv,
    int32_t &selected_index, float &selected_score,
    int32_t * err_out = nullptr) noexcept {
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

  if constexpr (wmode == window_mode::streamed) {
    bind_streamed_output_views(backend);
  }
  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    if (!run_layer<mode, route, lanes, wmode>(backend, layer, kv, position,
                                              err_out)) {
      if constexpr (wmode == window_mode::streamed) {
        reset_stream_block_views(backend);
      }
      return false;
    }
  }
  backend.kv_cache_tokens = position + 1;
  if constexpr (wmode == window_mode::streamed) {
    // The argmax output stage runs on the raw output view the streamed route
    // was classified for; the resident views are restored after.
    const bool argmax_ok = compute_logits_preselected_argmax<argmax_route>(
        backend, selected_index, selected_score);
    reset_stream_block_views(backend);
    return argmax_ok;
  } else {
    return compute_logits_preselected_argmax<argmax_route>(
        backend, selected_index, selected_score);
  }
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          scalar_argmax_route argmax_route,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_decode_preselected_argmax(
    native_backend &backend,
    const emel::graph::processor::event::execute &request,
    int32_t &selected_index, float &selected_score,
    int32_t * err_out = nullptr) noexcept {
  return run_decode_preselected_argmax<mode, route, argmax_route, lanes, wmode>(
      backend, request, identity_kv_addressing(), selected_index, selected_score, err_out);
}

}  // namespace

inline emel::error::type prepare(
    native_backend & backend,
    const emel::model::data & model_data,
    const int32_t kv_block_tokens = emel::memory::view::DEFAULT_BLOCK_TOKENS) noexcept {
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
  backend.kv_block_tokens = kv_block_tokens;
  backend.kv_positions_capacity =
      emel::memory::view::positions_capacity_for(kv_block_tokens, backend.n_ctx);
  if (backend.kv_positions_capacity < backend.n_ctx) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
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
  const bool lfm2_runtime = is_lfm2_runtime(backend);
  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    emel::model::llama::detail::block_view block = {};
    if (emel::model::llama::detail::lookup_block_view(backend.execution, layer, block) !=
        emel::error::cast(emel::model::loader::error::none)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    auto & weights = backend.blocks[static_cast<size_t>(layer)];
    weights.uses_attention = block.uses_attention;
    weights.attention_rope_pairing = {
        model_data.params.rope_pair_x0_stride,
        model_data.params.rope_pair_x1_stride,
        model_data.params.rope_pair_x1_offset,
        model_data.params.rope_pair_x1_half_rot_offset,
    };
    const bool common_ok =
        dequantize_tensor_vector(*block.attention_norm.tensor, weights.attention_norm) &&
        dequantize_tensor_vector(*block.feed_forward_norm.tensor, weights.feed_forward_norm) &&
        bind_tensor_rows(*block.feed_forward_gate.tensor, weights.feed_forward_gate) &&
        bind_tensor_rows(*block.feed_forward_down.tensor, weights.feed_forward_down) &&
        bind_tensor_rows(*block.feed_forward_up.tensor, weights.feed_forward_up);
    if (!common_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    if (weights.uses_attention) {
      const bool attention_ok =
          bind_tensor_rows(*block.attention_q.tensor, weights.attention_q) &&
          bind_tensor_rows(*block.attention_k.tensor, weights.attention_k) &&
          bind_tensor_rows(*block.attention_v.tensor, weights.attention_v) &&
          bind_tensor_rows(*block.attention_output.tensor, weights.attention_output) &&
          (!requires_attention_qk_norm(backend, weights) ||
           dequantize_tensor_vector(*block.attention_q_norm.tensor, weights.attention_q_norm)) &&
          (!requires_attention_qk_norm(backend, weights) ||
           dequantize_tensor_vector(*block.attention_k_norm.tensor, weights.attention_k_norm));
      if (!attention_ok) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      continue;
    }

    if (!lfm2_runtime ||
        !bind_tensor_rows(*block.shortconv_in_proj.tensor, weights.shortconv_in_proj) ||
        !bind_tensor_rows(*block.shortconv_out_proj.tensor, weights.shortconv_out_proj) ||
        block.shortconv_conv.tensor == nullptr ||
        static_cast<uint8_t>(block.shortconv_conv.tensor->type) !=
            emel::kernel::detail::dtype_f32 ||
        block.shortconv_conv.tensor->n_dims != 2) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    const int32_t shortconv_rows = static_cast<int32_t>(block.shortconv_conv.tensor->dims[1]);
    const int32_t shortconv_cols = static_cast<int32_t>(block.shortconv_conv.tensor->dims[0]);
    if (shortconv_rows <= 0 || shortconv_cols <= 0) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    weights.shortconv_conv.resize(
        static_cast<size_t>(shortconv_rows) * static_cast<size_t>(shortconv_cols));
    std::memcpy(
        weights.shortconv_conv.data(),
        block.shortconv_conv.tensor->data,
        weights.shortconv_conv.size() * sizeof(float));
  }

  if (!prepare_block_native_matrices(backend) ||
      !prepare_q8_input_workspace(backend) ||
      !prepare_q8_input_chunk4_workspace(backend) ||
      !prepare_q8_input_chunk8_workspace(backend) ||
      !prepare_packed_q8_0_input_workspace(backend) ||
      !prepare_packed_q8_0_chunk4_input_workspace(backend)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const int32_t declared_key_length = model_data.params.attention_key_length;
  const int32_t declared_key_length_swa = model_data.params.attention_key_length_swa;
  const int32_t declared_value_length = model_data.params.attention_value_length;
  const int32_t declared_value_length_swa = model_data.params.attention_value_length_swa;
  const block_weights * attention_block = first_attention_block(backend);
  if (attention_block == nullptr) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  backend.n_rep = backend.n_head / backend.n_head_kv;
  backend.shortconv_kernel_size = model_data.params.shortconv_l_cache;
  backend.shortconv_state_size =
      backend.shortconv_kernel_size > 0 ? backend.shortconv_kernel_size - 1 : 0;
  backend.max_q_dim = 0;
  backend.max_kv_dim = 0;
  backend.max_ffn_dim = 0;
  backend.layer_cache_offsets.resize(static_cast<size_t>(backend.n_layer));
  backend.flash_layer_cache_offsets.resize(static_cast<size_t>(backend.n_layer));
  size_t cache_offset = 0u;
  size_t flash_cache_offset = 0u;

  for (int32_t layer = 0; layer < backend.n_layer; ++layer) {
    auto & block = backend.blocks[static_cast<size_t>(layer)];
    if (static_cast<int32_t>(block.attention_norm.size()) != backend.n_embd ||
        static_cast<int32_t>(block.feed_forward_norm.size()) != backend.n_embd ||
        block.feed_forward_gate.rows <= 0 ||
        block.feed_forward_gate.cols != backend.n_embd ||
        block.feed_forward_up.rows != block.feed_forward_gate.rows ||
        block.feed_forward_up.cols != backend.n_embd ||
        block.feed_forward_down.rows != backend.n_embd ||
        block.feed_forward_down.cols != block.feed_forward_gate.rows) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    backend.max_ffn_dim = std::max(backend.max_ffn_dim, block.feed_forward_gate.rows);

    if (block.uses_attention) {
      block.attention_q_dim = block.attention_q.rows;
      block.attention_kv_dim = block.attention_k.rows;
      if (block.attention_q.cols != backend.n_embd ||
          block.attention_q.rows <= 0 ||
          (block.attention_q.rows % backend.n_head) != 0 ||
          block.attention_k.cols != backend.n_embd ||
          block.attention_k.rows <= 0 ||
          (block.attention_k.rows % backend.n_head_kv) != 0 ||
          block.attention_v.cols != backend.n_embd ||
          block.attention_v.rows != block.attention_k.rows ||
          block.attention_output.cols != block.attention_q.rows ||
          block.attention_output.rows != backend.n_embd) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }

      block.attention_head_dim = block.attention_q.rows / backend.n_head;
      block.attention_head_dim_kv = block.attention_k.rows / backend.n_head_kv;
      const bool gemma4_sliding_attention = is_gemma4_sliding_attention_layer(backend, layer);
      const int32_t expected_key_length =
          gemma4_sliding_attention && declared_key_length_swa > 0 ? declared_key_length_swa
                                                                  : declared_key_length;
      const int32_t expected_value_length =
          gemma4_sliding_attention && declared_value_length_swa > 0 ? declared_value_length_swa
                                                                    : declared_value_length;
      const int32_t expected_rope_dim =
          gemma4_sliding_attention && model_data.params.n_rot_swa > 0 ? model_data.params.n_rot_swa
                                                                      : model_data.params.n_rot;
      block.attention_rope_dim =
          expected_rope_dim > 0 ? expected_rope_dim : block.attention_head_dim;
      block.attention_rope_freq_base =
          gemma4_sliding_attention && model_data.params.rope_freq_base_swa > 0.0f
              ? model_data.params.rope_freq_base_swa
              : backend.rope_freq_base;
      const bool qk_norm_required = requires_attention_qk_norm(backend, block);
      if (block.attention_head_dim <= 0 ||
          block.attention_head_dim_kv <= 0 ||
          (expected_key_length > 0 && block.attention_head_dim != expected_key_length) ||
          (expected_value_length > 0 && block.attention_head_dim_kv != expected_value_length) ||
          (qk_norm_required &&
           static_cast<int32_t>(block.attention_q_norm.size()) != block.attention_head_dim) ||
          (qk_norm_required &&
           static_cast<int32_t>(block.attention_k_norm.size()) != block.attention_head_dim_kv)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }

      backend.max_q_dim = std::max(backend.max_q_dim, block.attention_q_dim);
      backend.max_kv_dim = std::max(backend.max_kv_dim, block.attention_kv_dim);
      backend.layer_cache_offsets[static_cast<size_t>(layer)] = cache_offset;
      cache_offset += static_cast<size_t>(backend.kv_positions_capacity) *
                      static_cast<size_t>(block.attention_kv_dim);
      backend.flash_layer_cache_offsets[static_cast<size_t>(layer)] = flash_cache_offset;
      flash_cache_offset += static_cast<size_t>(backend.n_head_kv) *
                            static_cast<size_t>(backend.kv_positions_capacity) *
                            static_cast<size_t>(block.attention_head_dim_kv);
      continue;
    }

    if (!lfm2_runtime ||
        backend.shortconv_kernel_size <= 1 ||
        block.shortconv_in_proj.cols != backend.n_embd ||
        block.shortconv_in_proj.rows != 3 * backend.n_embd ||
        block.shortconv_out_proj.cols != backend.n_embd ||
        block.shortconv_out_proj.rows != backend.n_embd ||
        static_cast<int32_t>(block.shortconv_conv.size()) !=
            backend.shortconv_kernel_size * backend.n_embd) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  backend.head_dim = backend.max_q_dim > 0 ? backend.max_q_dim / backend.n_head : 0;
  backend.head_dim_kv = backend.max_kv_dim > 0 ? backend.max_kv_dim / backend.n_head_kv : 0;
  if (backend.max_q_dim <= 0 || backend.max_kv_dim <= 0 || backend.n_rep <= 0) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  if (backend.max_ffn_dim <= 0) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  backend.key_cache.resize(cache_offset);
  backend.value_cache.resize(cache_offset);
  backend.flash_key_cache.resize(flash_cache_offset);
  backend.flash_value_cache.resize(flash_cache_offset);
  backend.recurrent_shortconv_cache.resize(
      static_cast<size_t>(backend.n_layer) *
      static_cast<size_t>(backend.shortconv_state_size) *
      static_cast<size_t>(backend.n_embd));
  backend.bound_logits.resize(static_cast<size_t>(backend.n_vocab));
  backend.bound_tokens.resize(static_cast<size_t>(backend.n_ctx));
  backend.bound_positions.resize(static_cast<size_t>(backend.n_ctx));
  backend.hidden.resize(static_cast<size_t>(backend.n_embd));
  backend.hidden_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(backend.n_embd));
  backend.hidden_chunk8.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(backend.n_embd));
  backend.norm.resize(static_cast<size_t>(backend.n_embd));
  backend.norm_chunk4.resize(backend.hidden_chunk4.size());
  backend.norm_chunk8.resize(backend.hidden_chunk8.size());
  backend.shortconv_bcx.resize(static_cast<size_t>(3 * backend.n_embd));
  backend.shortconv_bx.resize(static_cast<size_t>(backend.n_embd));
  backend.shortconv_conv_out.resize(static_cast<size_t>(backend.n_embd));
  backend.shortconv_bcx_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) *
      static_cast<size_t>(3 * backend.n_embd));
  backend.shortconv_conv_out_chunk4.resize(backend.hidden_chunk4.size());
  backend.shortconv_bcx_chunk8.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) *
      static_cast<size_t>(3 * backend.n_embd));
  backend.shortconv_conv_out_chunk8.resize(backend.hidden_chunk8.size());
  backend.q.resize(static_cast<size_t>(backend.max_q_dim));
  backend.q_attn.resize(static_cast<size_t>(backend.max_q_dim));
  backend.q_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(backend.max_q_dim));
  backend.q_chunk8.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(backend.max_q_dim));
  backend.k.resize(static_cast<size_t>(backend.max_kv_dim));
  backend.k_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) * static_cast<size_t>(backend.max_kv_dim));
  backend.k_chunk8.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) * static_cast<size_t>(backend.max_kv_dim));
  backend.v.resize(static_cast<size_t>(backend.max_kv_dim));
  backend.v_chunk4.resize(backend.k_chunk4.size());
  backend.v_chunk8.resize(backend.k_chunk8.size());
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs_rounded.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_value_column.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(static_cast<size_t>(backend.max_q_dim));
  backend.attn_ctx_chunk4.resize(backend.q_chunk4.size());
  backend.attn_ctx_chunk8.resize(backend.q_chunk8.size());
  backend.projected.resize(static_cast<size_t>(backend.n_embd));
  backend.projected_chunk4.resize(backend.hidden_chunk4.size());
  backend.projected_chunk8.resize(backend.hidden_chunk8.size());
  backend.gate.resize(static_cast<size_t>(backend.max_ffn_dim));
  backend.gate_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) *
      static_cast<size_t>(backend.max_ffn_dim));
  backend.gate_chunk8.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) *
      static_cast<size_t>(backend.max_ffn_dim));
  backend.up.resize(static_cast<size_t>(backend.max_ffn_dim));
  backend.up_chunk4.resize(
      static_cast<size_t>(k_prefill_q8_chunk_rows) *
      static_cast<size_t>(backend.max_ffn_dim));
  backend.up_chunk8.resize(
      static_cast<size_t>(k_prefill_q8_chunk8_rows) *
      static_cast<size_t>(backend.max_ffn_dim));
  backend.ffn_hidden.resize(static_cast<size_t>(backend.max_ffn_dim));
  backend.ffn_hidden_chunk4.resize(backend.gate_chunk4.size());
  backend.ffn_hidden_chunk8.resize(backend.gate_chunk8.size());
  build_lifecycle(backend);

  for (auto & lane_kernel : backend.lane_kernels) {
    lane_kernel.set_kind(backend.kernel_kind);
  }
  // One-time worker-thread construction for the parallel matmul lanes; the
  // pool is engaged here so no thread creation ever happens during dispatch.
  backend.lane_pool.emplace();

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

inline emel::text::generator::compute_io & bind_compute_io(
    const emel::graph::processor::event::execute & request) noexcept;
inline native_backend & bind_native_backend(
    const emel::graph::processor::event::execute & request) noexcept;

inline bool validate_guarded_compute(const emel::graph::processor::event::execute & request,
                                     int32_t * err_out) noexcept {
  (void)request;
  (void)err_out;
  return true;
}

inline bool validate_guarded_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  (void)request;
  (void)err_out;
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

inline emel::text::generator::compute_io & bind_compute_io(
    const emel::graph::processor::event::execute & request) noexcept {
  return *static_cast<emel::text::generator::compute_io *>(request.compute_ctx);
}

inline native_backend & bind_native_backend(
    const emel::graph::processor::event::execute & request) noexcept {
  return *static_cast<native_backend *>(bind_compute_io(request).backend_ctx);
}

inline bool bind_guarded_inputs(const emel::graph::processor::event::execute & request,
                                int32_t * err_out) noexcept {
  (void)err_out;
  auto & io = bind_compute_io(request);
  auto & backend = bind_native_backend(request);
  std::copy_n(io.token_ids, io.token_count, backend.bound_tokens.begin());
  std::copy_n(request.positions, request.positions_count, backend.bound_positions.begin());
  backend.bound_token_count = io.token_count;
  backend.bound_position_count = request.positions_count;
  backend.bound_ready = true;
  return true;
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          step_kind expected_kind,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_kernel_scalar_mode(const emel::graph::processor::event::execute & request,
                                   int32_t * err_out) noexcept {
  auto & backend = bind_native_backend(request);
  const auto kv = kv_addressing_from_request(request);
  if constexpr (expected_kind == step_kind::prefill) {
    (void)err_out;
    return run_prefill<mode, route>(backend, kv);
  } else {
    return run_decode<mode, route, lanes, wmode>(backend, request, kv, err_out);
  }
}

template <emel::text::generator::attention_mode mode, scalar_matmul_route route,
          scalar_argmax_route argmax_route, step_kind expected_kind,
          matmul_lane_mode lanes = matmul_lane_mode::serial,
          window_mode wmode = window_mode::resident>
inline bool run_kernel_scalar_preselected_argmax_mode(
    const emel::graph::processor::event::execute &request,
    int32_t * err_out) noexcept {
  auto & io = bind_compute_io(request);
  auto & backend = bind_native_backend(request);
  const auto kv = kv_addressing_from_request(request);
  if constexpr (expected_kind == step_kind::prefill) {
    return run_prefill_preselected_argmax<mode, route, argmax_route>(
        backend, kv, *io.selected_token_out, *io.selected_score_out);
  } else {
    // The decode branch forwards the graph processor's err pointer: streamed
    // instantiations report acquire failures through it (run_layer writes
    // unconditionally under the always-valid-pointer contract).
    return run_decode_preselected_argmax<mode, route, argmax_route, lanes,
                                         wmode>(
        backend, request, kv, *io.selected_token_out, *io.selected_score_out,
        err_out);
  }
}

inline bool run_kernel_flash_prefill_scalar_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::packed_q8_0,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_native_quantized(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_native_quantized_q8_k_logits(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::packed_q8_0,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::q8_k,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_native_quantized(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_native_quantized_q8_k_logits(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_decode_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::packed_q8_0,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_packed_q8_0_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::packed_q8_0,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_flash_decode_q8_k_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_flash_decode_native_quantized(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_native_quantized_q8_k_logits(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_native_quantized_q8_k_logits_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_flash_decode_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      step_kind::decode>(request, err_out);
}

// Streamed siblings of the flash scalar decode routes: identical arithmetic,
// weights acquired per layer from the tensor window actor.
inline bool run_kernel_flash_decode_kernel_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_flash_decode_native_quantized_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::packed_q8_0,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_native_quantized(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_native_quantized_q8_k_logits(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_nonflash_decode_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::packed_q8_0,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::q8_k,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_native_quantized(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_native_quantized_q8_k_logits(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel,
      step_kind::decode>(request, err_out);
}

// Streamed variants of the serial nonflash decode routes: identical compute
// with window_mode::streamed threading through run_layer's per-layer slot
// acquire, so an active tensor window streams on the nonflash decode path
// exactly as it does on the flash path.
inline bool run_kernel_nonflash_decode_packed_q8_0_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::packed_q8_0,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_nonflash_decode_q8_k_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_nonflash_decode_native_quantized_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_nonflash_decode_native_quantized_q8_k_logits_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized_q8_k_logits,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_nonflash_decode_kernel_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_kernel_prefill_chunk8_q8_k_mode(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  (void)err_out;
  return run_prefill_chunk8_q8_k<mode, lanes>(
      bind_native_backend(request), kv_addressing_from_request(request));
}

inline bool run_kernel_flash_prefill_chunk8_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk8_q8_k_mode<emel::text::generator::attention_mode::flash>(
      request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk8_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk8_q8_k_mode<emel::text::generator::attention_mode::nonflash>(
      request, err_out);
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_kernel_prefill_chunk4_mode(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  (void)err_out;
  return run_prefill_chunk4<mode, route, lanes>(
      bind_native_backend(request), kv_addressing_from_request(request));
}

inline bool run_kernel_flash_prefill_chunk4_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::packed_q8_0>(
      request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk4_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<
      emel::text::generator::attention_mode::nonflash,
      chunk4_rhs_route::packed_q8_0>(
      request, err_out);
}

inline bool run_kernel_flash_prefill_chunk4_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::q8_k>(
      request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk4_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<
      emel::text::generator::attention_mode::nonflash,
      chunk4_rhs_route::q8_k>(
      request, err_out);
}

inline bool run_kernel_flash_prefill_parallel_chunk8_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk8_q8_k_mode<
      emel::text::generator::attention_mode::flash,
      matmul_lane_mode::parallel>(
      request, err_out);
}

inline bool run_kernel_flash_prefill_parallel_chunk4_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::packed_q8_0,
      matmul_lane_mode::parallel>(
      request, err_out);
}

inline bool run_kernel_flash_prefill_parallel_chunk4_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::q8_k,
      matmul_lane_mode::parallel>(
      request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_preselected_argmax_native_quantized_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::q8_k,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_preselected_argmax_native_quantized_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::kernel,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_prefill_scalar_preselected_argmax_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      scalar_argmax_route::kernel,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_preselected_argmax_native_quantized_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::q8_k,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_preselected_argmax_native_quantized_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::kernel,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_scalar_preselected_argmax_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel,
      scalar_argmax_route::kernel,
      step_kind::prefill>(request, err_out);
}

inline bool run_kernel_flash_decode_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_preselected_argmax_native_quantized_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::q8_k,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_preselected_argmax_native_quantized_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::kernel,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_flash_decode_preselected_argmax_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      scalar_argmax_route::kernel,
      step_kind::decode>(request, err_out);
}

// Streamed variants of the serial preselected routes: identical compute with
// window_mode::streamed threading through run_layer's per-layer slot acquire,
// so the tensor-window lane consumes slot bytes on the preselected family too.
inline bool run_kernel_flash_decode_preselected_argmax_q8_k_streamed(
    const emel::graph::processor::event::execute &request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash, scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k, step_kind::decode, matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool
run_kernel_flash_decode_preselected_argmax_native_quantized_q8_k_streamed(
    const emel::graph::processor::event::execute &request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized, scalar_argmax_route::q8_k,
      step_kind::decode, matmul_lane_mode::serial, window_mode::streamed>(
      request, err_out);
}

inline bool
run_kernel_flash_decode_preselected_argmax_native_quantized_kernel_streamed(
    const emel::graph::processor::event::execute &request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized, scalar_argmax_route::kernel,
      step_kind::decode, matmul_lane_mode::serial, window_mode::streamed>(
      request, err_out);
}

inline bool run_kernel_flash_decode_preselected_argmax_kernel_streamed(
    const emel::graph::processor::event::execute &request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash, scalar_matmul_route::kernel,
      scalar_argmax_route::kernel, step_kind::decode, matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_preselected_argmax_native_quantized_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_preselected_argmax_native_quantized_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::kernel,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_decode_parallel_preselected_argmax_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      scalar_matmul_route::kernel,
      scalar_argmax_route::kernel,
      step_kind::decode,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_nonflash_decode_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_preselected_argmax_native_quantized_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::q8_k,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_preselected_argmax_native_quantized_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::kernel,
      step_kind::decode>(request, err_out);
}

inline bool run_kernel_nonflash_decode_preselected_argmax_kernel(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel,
      scalar_argmax_route::kernel,
      step_kind::decode>(request, err_out);
}

// Streamed variants of the serial nonflash preselected routes (see the
// nonflash decode streamed wrappers above).
inline bool run_kernel_nonflash_decode_preselected_argmax_q8_k_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::q8_k,
      scalar_argmax_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool
run_kernel_nonflash_decode_preselected_argmax_native_quantized_q8_k_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::q8_k,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool
run_kernel_nonflash_decode_preselected_argmax_native_quantized_kernel_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::native_quantized,
      scalar_argmax_route::kernel,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

inline bool run_kernel_nonflash_decode_preselected_argmax_kernel_streamed(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_scalar_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      scalar_matmul_route::kernel,
      scalar_argmax_route::kernel,
      step_kind::decode,
      matmul_lane_mode::serial,
      window_mode::streamed>(request, err_out);
}

template <emel::text::generator::attention_mode mode,
          chunk4_rhs_route route,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_kernel_prefill_chunk4_preselected_argmax_mode(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  (void)err_out;
  auto & io = bind_compute_io(request);
  return run_prefill_chunk4_preselected_argmax<mode, route, lanes>(
      bind_native_backend(request), kv_addressing_from_request(request),
      *io.selected_token_out, *io.selected_score_out);
}

template <emel::text::generator::attention_mode mode,
          matmul_lane_mode lanes = matmul_lane_mode::serial>
inline bool run_kernel_prefill_chunk8_preselected_argmax_q8_k_mode(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  (void)err_out;
  auto & io = bind_compute_io(request);
  return run_prefill_chunk8_preselected_argmax_q8_k<mode, lanes>(
      bind_native_backend(request), kv_addressing_from_request(request),
      *io.selected_token_out, *io.selected_score_out);
}

inline bool run_kernel_flash_prefill_chunk8_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk8_preselected_argmax_q8_k_mode<
      emel::text::generator::attention_mode::flash>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk8_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk8_preselected_argmax_q8_k_mode<
      emel::text::generator::attention_mode::nonflash>(request, err_out);
}

inline bool run_kernel_flash_prefill_chunk4_preselected_argmax_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::packed_q8_0>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      chunk4_rhs_route::packed_q8_0>(request, err_out);
}

inline bool run_kernel_flash_prefill_chunk4_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::q8_k>(request, err_out);
}

inline bool run_kernel_nonflash_prefill_chunk4_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::text::generator::attention_mode::nonflash,
      chunk4_rhs_route::q8_k>(request, err_out);
}

inline bool run_kernel_flash_prefill_parallel_chunk8_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk8_preselected_argmax_q8_k_mode<
      emel::text::generator::attention_mode::flash,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_prefill_parallel_chunk4_preselected_argmax_packed_q8_0(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::packed_q8_0,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool run_kernel_flash_prefill_parallel_chunk4_preselected_argmax_q8_k(
    const emel::graph::processor::event::execute & request,
    int32_t * err_out) noexcept {
  return run_kernel_prefill_chunk4_preselected_argmax_mode<
      emel::text::generator::attention_mode::flash,
      chunk4_rhs_route::q8_k,
      matmul_lane_mode::parallel>(request, err_out);
}

inline bool extract_guarded_outputs(const emel::graph::processor::event::execute & request,
                                    int32_t * outputs_out,
                                    int32_t * err_out) noexcept {
  (void)err_out;
  auto & io = bind_compute_io(request);
  auto & backend = bind_native_backend(request);
  std::copy(backend.bound_logits.begin(), backend.bound_logits.end(), io.logits);
  for (int32_t idx = backend.n_vocab; idx < io.logits_capacity; ++idx) {
    io.logits[idx] = -1.0f;
  }
  *outputs_out = 1;
  return true;
}

inline bool extract_guarded_preselected_argmax(
    const emel::graph::processor::event::execute & request,
    int32_t * outputs_out,
    int32_t * err_out) noexcept {
  (void)request;
  (void)err_out;
  *outputs_out = 1;
  return true;
}

}  // namespace emel::text::generator::detail
