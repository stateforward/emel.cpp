#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// Keep this list aligned with `tmp/llama.cpp/ggml/include/ggml.h` (`enum ggml_op`),
// excluding sentinel entries (`NONE`, `COUNT`).
#define EMEL_KERNEL_OP_EVENT_LIST(X) \
  X(op_dup) \
  X(op_add) \
  X(op_add_id) \
  X(op_add1) \
  X(op_acc) \
  X(op_sub) \
  X(op_mul) \
  X(op_div) \
  X(op_sqr) \
  X(op_sqrt) \
  X(op_log) \
  X(op_sin) \
  X(op_cos) \
  X(op_sum) \
  X(op_sum_rows) \
  X(op_cumsum) \
  X(op_mean) \
  X(op_argmax) \
  X(op_count_equal) \
  X(op_repeat) \
  X(op_repeat_back) \
  X(op_concat) \
  X(op_silu_back) \
  X(op_norm) \
  X(op_rms_norm) \
  X(op_rms_norm_back) \
  X(op_group_norm) \
  X(op_l2_norm) \
  X(op_mul_mat) \
  X(op_mul_mat_id) \
  X(op_out_prod) \
  X(op_scale) \
  X(op_set) \
  X(op_cpy) \
  X(op_cont) \
  X(op_reshape) \
  X(op_view) \
  X(op_permute) \
  X(op_transpose) \
  X(op_get_rows) \
  X(op_get_rows_back) \
  X(op_set_rows) \
  X(op_diag) \
  X(op_diag_mask_inf) \
  X(op_diag_mask_zero) \
  X(op_soft_max) \
  X(op_soft_max_back) \
  X(op_rope) \
  X(op_rope_back) \
  X(op_clamp) \
  X(op_conv_transpose_1d) \
  X(op_im2col) \
  X(op_im2col_back) \
  X(op_im2col_3d) \
  X(op_conv_2d) \
  X(op_conv_3d) \
  X(op_conv_2d_dw) \
  X(op_conv_transpose_2d) \
  X(op_pool_1d) \
  X(op_pool_2d) \
  X(op_pool_2d_back) \
  X(op_upscale) \
  X(op_pad) \
  X(op_pad_reflect_1d) \
  X(op_roll) \
  X(op_arange) \
  X(op_timestep_embedding) \
  X(op_argsort) \
  X(op_top_k) \
  X(op_leaky_relu) \
  X(op_tri) \
  X(op_fill) \
  X(op_flash_attn_ext) \
  X(op_flash_attn_back) \
  X(op_ssm_conv) \
  X(op_ssm_scan) \
  X(op_win_part) \
  X(op_win_unpart) \
  X(op_get_rel_pos) \
  X(op_add_rel_pos) \
  X(op_rwkv_wkv6) \
  X(op_gated_linear_attn) \
  X(op_rwkv_wkv7) \
  X(op_solve_tri) \
  X(op_unary) \
  X(op_map_custom1) \
  X(op_map_custom2) \
  X(op_map_custom3) \
  X(op_custom) \
  X(op_cross_entropy_loss) \
  X(op_cross_entropy_loss_back) \
  X(op_opt_step_adamw) \
  X(op_opt_step_sgd) \
  X(op_glu)

namespace emel::kernel::event {

enum class dtype : uint8_t;
enum class unary_subop : uint8_t;

#define EMEL_KERNEL_FORWARD_DECLARE_EVENT(op_name) \
  struct op_name;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_FORWARD_DECLARE_EVENT)
#undef EMEL_KERNEL_FORWARD_DECLARE_EVENT

}  // namespace emel::kernel::event

namespace emel::kernel {

template <class event_type>
struct is_op_event : std::false_type {};

#define EMEL_KERNEL_MARK_OP_EVENT(op_name) \
  template <>                              \
  struct is_op_event<event::op_name> : std::true_type {};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_MARK_OP_EVENT)
#undef EMEL_KERNEL_MARK_OP_EVENT

template <class event_type>
inline constexpr bool is_op_event_v = is_op_event<event_type>::value;

}  // namespace emel::kernel

namespace emel::kernel::detail {

inline constexpr uint8_t dtype_f32 = 0;
inline constexpr uint8_t dtype_q4_0 = 2;

inline uint64_t select_u64(const bool choose_true,
                           const uint64_t true_value,
                           const uint64_t false_value) noexcept {
  const uint64_t mask = static_cast<uint64_t>(0) - static_cast<uint64_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline bool select_bool(const bool choose_true,
                        const bool true_value,
                        const bool false_value) noexcept {
  const std::array<bool, 2> values{false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

template <class dtype_type>
inline uint8_t dtype_code(const dtype_type type) noexcept {
  return static_cast<uint8_t>(type);
}

inline bool is_supported_dtype(const uint8_t code) noexcept {
  return code == dtype_f32;
}

inline size_t dtype_size_bytes(const uint8_t code) noexcept {
  const std::array<size_t, 2> size_candidates = {0u, 4u};
  return size_candidates[static_cast<size_t>(code == dtype_f32)];
}

template <class tensor_type>
inline uint64_t tensor_element_count(const tensor_type & tensor) noexcept {
  uint64_t count = 1;
  for (size_t i = 0; i < 4; ++i) {
    count *= tensor.ne[i];
  }
  return count;
}

template <class tensor_type>
inline uint64_t tensor_stride_bytes(const tensor_type & tensor, const size_t dim) noexcept {
  uint64_t stride = dtype_size_bytes(dtype_code(tensor.type));
  for (size_t i = 0; i < dim; ++i) {
    stride *= tensor.ne[i];
  }
  const std::array<uint64_t, 2> candidates{stride, tensor.nb[dim]};
  return candidates[static_cast<size_t>(tensor.nb[0] != 0)];
}

template <class tensor_type>
inline bool has_valid_tensor_layout(const tensor_type & tensor) noexcept {
  const uint64_t elem_size = dtype_size_bytes(dtype_code(tensor.type));
  const bool elem_valid = elem_size != 0u;
  const bool explicit_stride = tensor.nb[0] != 0u;
  const bool aligned_stride =
      explicit_stride && tensor.nb[0] >= elem_size && (tensor.nb[0] % elem_size) == 0u;

  bool dims_valid = true;
  for (size_t i = 0; i < 4; ++i) {
    const bool invalid_dim = tensor.ne[i] > 1 && tensor.nb[i] == 0;
    dims_valid = dims_valid && !invalid_dim;
  }

  return elem_valid && (!explicit_stride || (aligned_stride && dims_valid));
}

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type & tensor) noexcept {
  const bool valid_layout = has_valid_tensor_layout(tensor);
  uint64_t expected = dtype_size_bytes(dtype_code(tensor.type));
  bool matches = true;
  for (size_t i = 0; i < 4; ++i) {
    matches = matches && tensor_stride_bytes(tensor, i) == expected;
    expected *= tensor.ne[i];
  }
  return valid_layout && matches;
}

template <class tensor_type>
inline size_t tensor_offset_bytes(const tensor_type & tensor, const uint64_t idx) noexcept {
  uint64_t remaining = idx;
  size_t offset = 0;
  bool dims_active = true;
  for (size_t d = 0; d < 4; ++d) {
    const bool dim_non_zero = tensor.ne[d] != 0u;
    const bool step_active = dims_active && dim_non_zero;
    const uint64_t dim = select_u64(step_active, tensor.ne[d], 1u);
    const uint64_t coord = remaining % dim;
    const uint64_t stride = tensor_stride_bytes(tensor, d);
    const uint64_t offset_step = coord * stride;
    offset += static_cast<size_t>(select_u64(step_active, offset_step, 0u));
    remaining = select_u64(step_active, remaining / dim, remaining);
    dims_active = dims_active && dim_non_zero;
  }
  return offset;
}

template <class tensor_type>
inline size_t tensor_offset_bytes(const tensor_type & tensor,
                                  const uint64_t i0,
                                  const uint64_t i1,
                                  const uint64_t i2 = 0,
                                  const uint64_t i3 = 0) noexcept {
  return static_cast<size_t>(i0 * tensor_stride_bytes(tensor, 0) +
                             i1 * tensor_stride_bytes(tensor, 1) +
                             i2 * tensor_stride_bytes(tensor, 2) +
                             i3 * tensor_stride_bytes(tensor, 3));
}

template <class request_type>
inline constexpr bool requires_src1_v =
    std::is_same_v<request_type, event::op_add> ||
    std::is_same_v<request_type, event::op_add_id> ||
    std::is_same_v<request_type, event::op_add1> ||
    std::is_same_v<request_type, event::op_acc> ||
    std::is_same_v<request_type, event::op_sub> ||
    std::is_same_v<request_type, event::op_mul> ||
    std::is_same_v<request_type, event::op_div> ||
    std::is_same_v<request_type, event::op_mul_mat>;

template <class request_type>
inline bool has_required_src0(const request_type & request) noexcept {
  return request.src0.data != nullptr &&
         is_supported_dtype(dtype_code(request.src0.type)) &&
         has_valid_tensor_layout(request.src0) &&
         tensor_element_count(request.src0) > 0;
}

template <class request_type>
inline bool has_required_src1(const request_type & request) noexcept {
  if constexpr (!requires_src1_v<request_type>) {
    return true;
  }
  return request.src1.data != nullptr &&
         is_supported_dtype(dtype_code(request.src1.type)) &&
         has_valid_tensor_layout(request.src1) &&
         tensor_element_count(request.src1) > 0;
}

template <class request_type>
inline bool has_required_dst(const request_type & request) noexcept {
  return request.dst.data != nullptr &&
         is_supported_dtype(dtype_code(request.dst.type)) &&
         has_valid_tensor_layout(request.dst) &&
         tensor_element_count(request.dst) > 0;
}

template <class request_type>
inline bool validate_dispatch_request(const request_type & request) noexcept {
  const bool has_required_buffers =
      has_required_src0(request) && has_required_src1(request) && has_required_dst(request);
  const bool has_valid_threading = request.ith == 0 && request.nth == 1;
  const bool has_valid_params = request.op_params_size <= request.op_params.size();
  return has_required_buffers && has_valid_threading && has_valid_params;
}

template <class tensor_type>
inline float read_f32(const tensor_type & tensor, const uint64_t idx) noexcept {
  const bool dense = is_dense_contiguous(tensor);
  const float * data = static_cast<const float *>(tensor.data);
  const char * base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  const char *dense_src = reinterpret_cast<const char *>(data + idx);
  const char *sparse_src = base + offset;
  const std::array<const char *, 2> srcs{sparse_src, dense_src};
  float out = 0.0f;
  std::memcpy(&out, srcs[static_cast<size_t>(dense)], sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32(const tensor_type & tensor, const uint64_t idx, const float value) noexcept {
  const bool dense = is_dense_contiguous(tensor);
  float * data = static_cast<float *>(tensor.data);
  char * base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  char *dense_dst = reinterpret_cast<char *>(data + idx);
  char *sparse_dst = base + offset;
  const std::array<char *, 2> dsts{sparse_dst, dense_dst};
  std::memcpy(dsts[static_cast<size_t>(dense)], &value, sizeof(value));
}

template <class tensor_type>
inline float read_f32_at(const tensor_type & tensor, const uint64_t i0, const uint64_t i1,
                         const uint64_t i2 = 0, const uint64_t i3 = 0) noexcept {
  float out = 0.0f;
  const char * base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, i0, i1, i2, i3);
  std::memcpy(&out, base + offset, sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32_at(const tensor_type & tensor, const uint64_t i0, const uint64_t i1,
                         const float value, const uint64_t i2 = 0,
                         const uint64_t i3 = 0) noexcept {
  char * base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, i0, i1, i2, i3);
  std::memcpy(base + offset, &value, sizeof(value));
}

template <class request_type>
inline bool run_copy(const request_type & request) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool shape_ok = count == tensor_element_count(request.src0);
  const bool dense = shape_ok && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count = count * static_cast<uint64_t>(shape_ok && !dense);

  const float *src_dense = static_cast<const float *>(request.src0.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = src_dense[i];
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i, read_f32(request.src0, i));
  }
  return shape_ok;
}

template <class request_type, class op_type>
inline bool run_binary(const request_type & request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool incompatible_shape =
      count != tensor_element_count(request.src0) || count != tensor_element_count(request.src1);
  const bool compatible = !incompatible_shape;

  const bool dense = compatible &&
      is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count = count * static_cast<uint64_t>(compatible && !dense);

  const float *lhs_dense = static_cast<const float *>(request.src0.data);
  const float *rhs_dense = static_cast<const float *>(request.src1.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = op(lhs_dense[i], rhs_dense[i]);
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i, op(read_f32(request.src0, i), read_f32(request.src1, i)));
  }
  return compatible;
}

template <class request_type, class op_type>
inline bool run_unary(const request_type & request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool shape_ok = count == tensor_element_count(request.src0);
  const bool dense = shape_ok && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count = count * static_cast<uint64_t>(shape_ok && !dense);

  const float *src_dense = static_cast<const float *>(request.src0.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = op(src_dense[i]);
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i, op(read_f32(request.src0, i)));
  }
  return shape_ok;
}

inline constexpr uint8_t unary_subop_abs = 0u;
inline constexpr uint8_t unary_subop_neg = 2u;
inline constexpr uint8_t unary_subop_relu = 6u;
inline constexpr uint8_t unary_subop_exp = 13u;

template <uint8_t subop_code, class request_type>
inline void execute_scalar_unary_subop_unchecked(const request_type & request) noexcept {
  if constexpr (subop_code == unary_subop_abs) {
    (void) run_unary(request, [](const float v) { return std::fabs(v); });
  } else if constexpr (subop_code == unary_subop_neg) {
    (void) run_unary(request, [](const float v) { return -v; });
  } else if constexpr (subop_code == unary_subop_relu) {
    (void) run_unary(request, [](const float v) { return std::max(0.0f, v); });
  } else if constexpr (subop_code == unary_subop_exp) {
    (void) run_unary(request, [](const float v) { return std::exp(v); });
  }
}

template <class dispatch_event_type, class context_type, class mark_done_type,
          ::emel::kernel::event::unary_subop subop>
struct exec_scalar_unary_op {
  void operator()(const dispatch_event_type & ev, context_type & ctx) const noexcept {
    execute_scalar_unary_subop_unchecked<static_cast<uint8_t>(subop)>(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class simd_guard_type,
          class unary_subop_guard_type>
struct simd_unary_subop_guard {
  bool operator()(const dispatch_event_type & ev, const context_type & ctx) const noexcept {
    return simd_guard_type{}(ev, ctx) && unary_subop_guard_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class valid_guard_type,
          class unary_subop_guard_type>
struct valid_unary_subop_guard {
  bool operator()(const dispatch_event_type & ev, const context_type & ctx) const noexcept {
    return valid_guard_type{}(ev, ctx) && unary_subop_guard_type{}(ev, ctx);
  }
};

template <class request_type>
inline bool run_mul_mat(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool shape_mismatch =
      request.src1.ne[1] != k || request.dst.ne[0] != n || request.dst.ne[1] != m;
  const bool invalid_rank =
      request.src0.ne[2] != 1 || request.src0.ne[3] != 1 ||
      request.src1.ne[2] != 1 || request.src1.ne[3] != 1 ||
      request.dst.ne[2] != 1 || request.dst.ne[3] != 1;
  const bool valid = !(has_empty_dim || shape_mismatch || invalid_rank);
  const bool dense = valid &&
      is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
  const uint64_t dense_m = m * static_cast<uint64_t>(dense);
  const uint64_t sparse_m = m * static_cast<uint64_t>(valid && !dense);

  const float *a_dense = static_cast<const float *>(request.src0.data);
  const float *b_dense = static_cast<const float *>(request.src1.data);
  float *c_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < k; ++p) {
        acc += a_dense[i * k + p] * b_dense[p * n + j];
      }
      c_dense[i * n + j] = acc;
    }
  }

  for (uint64_t i = 0; i < sparse_m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < k; ++p) {
        acc += read_f32_at(request.src0, p, i) * read_f32_at(request.src1, j, p);
      }
      write_f32_at(request.dst, j, i, acc);
    }
  }

  return valid;
}

template <class request_type>
inline bool run_soft_max(const request_type & request) noexcept {
  const uint64_t width = request.src0.ne[0];
  const uint64_t count = tensor_element_count(request.src0);
  const bool invalid_shape = width == 0 || count == 0 || count % width != 0 ||
      count != tensor_element_count(request.dst);
  const bool valid = !invalid_shape;
  const uint64_t safe_width = select_u64(width != 0u, width, 1u);
  const uint64_t rows = (count / safe_width) * static_cast<uint64_t>(valid);

  const bool dense = valid && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_rows = rows * static_cast<uint64_t>(dense);
  const uint64_t sparse_rows = rows * static_cast<uint64_t>(!dense);

  const float *src_dense = static_cast<const float *>(request.src0.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t row = 0; row < dense_rows; ++row) {
    const uint64_t offset = row * width;
    float max_v = src_dense[offset];
    for (uint64_t i = 1; i < width; ++i) {
      max_v = std::max(max_v, src_dense[offset + i]);
    }

    float sum = 0.0f;
    for (uint64_t i = 0; i < width; ++i) {
      const float e = std::exp(src_dense[offset + i] - max_v);
      dst_dense[offset + i] = e;
      sum += e;
    }

    for (uint64_t i = 0; i < width; ++i) {
      dst_dense[offset + i] /= sum;
    }
  }

  for (uint64_t row = 0; row < sparse_rows; ++row) {
    const uint64_t offset = row * width;
    float max_v = read_f32(request.src0, offset);
    for (uint64_t i = 1; i < width; ++i) {
      max_v = std::max(max_v, read_f32(request.src0, offset + i));
    }

    float sum = 0.0f;
    for (uint64_t i = 0; i < width; ++i) {
      const float e = std::exp(read_f32(request.src0, offset + i) - max_v);
      write_f32(request.dst, offset + i, e);
      sum += e;
    }

    for (uint64_t i = 0; i < width; ++i) {
      write_f32(request.dst, offset + i, read_f32(request.dst, offset + i) / sum);
    }
  }

  return valid;
}

template <class request_type>
inline bool can_run_copy(const request_type & request) noexcept {
  return tensor_element_count(request.dst) == tensor_element_count(request.src0);
}

template <class request_type>
inline bool can_run_binary(const request_type & request) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  return count == tensor_element_count(request.src0) &&
         count == tensor_element_count(request.src1);
}

template <class request_type>
inline bool can_run_unary(const request_type & request) noexcept {
  return tensor_element_count(request.dst) == tensor_element_count(request.src0);
}

template <class request_type>
inline bool can_run_mul_mat(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool valid_shape = request.src1.ne[1] == k && request.dst.ne[0] == n &&
         request.dst.ne[1] == m && request.src0.ne[2] == 1 &&
         request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
         request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
         request.dst.ne[3] == 1;
  return !has_empty_dim && valid_shape;
}

template <class request_type>
inline bool can_run_soft_max(const request_type & request) noexcept {
  const uint64_t width = request.src0.ne[0];
  const uint64_t count = tensor_element_count(request.src0);
  return width != 0 && count != 0 && count % width == 0 &&
         count == tensor_element_count(request.dst);
}

template <class request_type>
inline bool can_run_unary_subop(const request_type & request) noexcept {
  const auto subop = static_cast<uint8_t>(request.subop);
  const bool supported_subop = subop == unary_subop_abs || subop == unary_subop_neg ||
                               subop == unary_subop_relu || subop == unary_subop_exp;
  return supported_subop && can_run_unary(request);
}

template <class request_type>
inline bool can_execute_scalar(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    return can_run_copy(request);
  } else if constexpr (std::is_same_v<request_type, event::op_add>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sub>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_div>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_log>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sin>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_cos>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    return can_run_mul_mat(request);
  } else if constexpr (std::is_same_v<request_type, event::op_soft_max>) {
    return can_run_soft_max(request);
  } else if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return false;
  }
  return false;
}

template <class request_type>
inline bool can_run_backend_request(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return can_run_unary_subop(request);
  }
  return can_execute_scalar(request);
}

template <class request_type>
inline void execute_scalar_unchecked(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    (void) run_copy(request);
  } else if constexpr (std::is_same_v<request_type, event::op_add>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs + rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_sub>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs - rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs * rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_div>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs / rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    (void) run_unary(request, [](const float v) { return v * v; });
  } else if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    (void) run_unary(request, [](const float v) { return std::sqrt(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_log>) {
    (void) run_unary(request, [](const float v) { return std::log(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_sin>) {
    (void) run_unary(request, [](const float v) { return std::sin(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_cos>) {
    (void) run_unary(request, [](const float v) { return std::cos(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    (void) run_mul_mat(request);
  } else if constexpr (std::is_same_v<request_type, event::op_soft_max>) {
    (void) run_soft_max(request);
  }
}

template <class request_type>
inline bool execute_scalar(const request_type & request) noexcept {
  const bool can_execute = can_execute_scalar(request);
  if (can_execute) {
    execute_scalar_unchecked(request);
  }
  return can_execute;
}

}  // namespace emel::kernel::detail
