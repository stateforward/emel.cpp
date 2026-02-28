#pragma once

#include <algorithm>
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

template <class dtype_type>
inline uint8_t dtype_code(const dtype_type type) noexcept {
  return static_cast<uint8_t>(type);
}

inline bool is_supported_dtype(const uint8_t code) noexcept {
  return code == dtype_f32;
}

inline size_t dtype_size_bytes(const uint8_t code) noexcept {
  return code == dtype_f32 ? 4u : 0u;
}

template <class tensor_type>
inline uint64_t tensor_element_count(const tensor_type & tensor) noexcept {
  uint64_t count = 1;
  for (size_t i = 0; i < 4; ++i) {
    if (tensor.ne[i] == 0) {
      return 0;
    }
    count *= tensor.ne[i];
  }
  return count;
}

template <class tensor_type>
inline uint64_t tensor_stride_bytes(const tensor_type & tensor, const size_t dim) noexcept {
  if (tensor.nb[0] != 0) {
    return tensor.nb[dim];
  }

  uint64_t stride = dtype_size_bytes(dtype_code(tensor.type));
  for (size_t i = 0; i < dim; ++i) {
    stride *= tensor.ne[i];
  }
  return stride;
}

template <class tensor_type>
inline bool has_valid_tensor_layout(const tensor_type & tensor) noexcept {
  const uint64_t elem_size = dtype_size_bytes(dtype_code(tensor.type));
  if (elem_size == 0) {
    return false;
  }

  if (tensor.nb[0] == 0) {
    return true;
  }

  if (tensor.nb[0] < elem_size || (tensor.nb[0] % elem_size) != 0) {
    return false;
  }

  for (size_t i = 0; i < 4; ++i) {
    if (tensor.ne[i] > 1 && tensor.nb[i] == 0) {
      return false;
    }
  }

  return true;
}

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type & tensor) noexcept {
  if (!has_valid_tensor_layout(tensor)) {
    return false;
  }

  uint64_t expected = dtype_size_bytes(dtype_code(tensor.type));
  for (size_t i = 0; i < 4; ++i) {
    if (tensor_stride_bytes(tensor, i) != expected) {
      return false;
    }
    expected *= tensor.ne[i];
  }
  return true;
}

template <class tensor_type>
inline size_t tensor_offset_bytes(const tensor_type & tensor, const uint64_t idx) noexcept {
  uint64_t remaining = idx;
  size_t offset = 0;
  for (size_t d = 0; d < 4; ++d) {
    const uint64_t dim = tensor.ne[d];
    if (dim == 0) {
      break;
    }
    const uint64_t coord = remaining % dim;
    remaining /= dim;
    offset += static_cast<size_t>(coord * tensor_stride_bytes(tensor, d));
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
  if (!has_required_src0(request) || !has_required_src1(request) || !has_required_dst(request)) {
    return false;
  }
  if (request.ith != 0 || request.nth != 1) {
    return false;
  }
  if (request.op_params_size > request.op_params.size()) {
    return false;
  }
  return true;
}

template <class tensor_type>
inline float read_f32(const tensor_type & tensor, const uint64_t idx) noexcept {
  if (is_dense_contiguous(tensor)) {
    const float * data = static_cast<const float *>(tensor.data);
    return data[idx];
  }

  float out = 0.0f;
  const char * base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  std::memcpy(&out, base + offset, sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32(const tensor_type & tensor, const uint64_t idx, const float value) noexcept {
  if (is_dense_contiguous(tensor)) {
    float * data = static_cast<float *>(tensor.data);
    data[idx] = value;
    return;
  }

  char * base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  std::memcpy(base + offset, &value, sizeof(value));
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
  if (count != tensor_element_count(request.src0)) {
    return false;
  }

  if (is_dense_contiguous(request.src0) && is_dense_contiguous(request.dst)) {
    const float * src = static_cast<const float *>(request.src0.data);
    float * dst = static_cast<float *>(request.dst.data);
    for (uint64_t i = 0; i < count; ++i) {
      dst[i] = src[i];
    }
    return true;
  }

  for (uint64_t i = 0; i < count; ++i) {
    write_f32(request.dst, i, read_f32(request.src0, i));
  }
  return true;
}

template <class request_type, class op_type>
inline bool run_binary(const request_type & request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  if (count != tensor_element_count(request.src0) ||
      count != tensor_element_count(request.src1)) {
    return false;
  }

  if (is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst)) {
    const float * lhs = static_cast<const float *>(request.src0.data);
    const float * rhs = static_cast<const float *>(request.src1.data);
    float * dst = static_cast<float *>(request.dst.data);
    for (uint64_t i = 0; i < count; ++i) {
      dst[i] = op(lhs[i], rhs[i]);
    }
    return true;
  }

  for (uint64_t i = 0; i < count; ++i) {
    write_f32(request.dst, i, op(read_f32(request.src0, i), read_f32(request.src1, i)));
  }
  return true;
}

template <class request_type, class op_type>
inline bool run_unary(const request_type & request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  if (count != tensor_element_count(request.src0)) {
    return false;
  }

  if (is_dense_contiguous(request.src0) && is_dense_contiguous(request.dst)) {
    const float * src = static_cast<const float *>(request.src0.data);
    float * dst = static_cast<float *>(request.dst.data);
    for (uint64_t i = 0; i < count; ++i) {
      dst[i] = op(src[i]);
    }
    return true;
  }

  for (uint64_t i = 0; i < count; ++i) {
    write_f32(request.dst, i, op(read_f32(request.src0, i)));
  }
  return true;
}

template <class request_type>
inline bool run_mul_mat(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  if (k == 0 || m == 0 || n == 0) {
    return false;
  }
  if (request.src1.ne[1] != k || request.dst.ne[0] != n || request.dst.ne[1] != m) {
    return false;
  }

  if (request.src0.ne[2] != 1 || request.src0.ne[3] != 1 ||
      request.src1.ne[2] != 1 || request.src1.ne[3] != 1 ||
      request.dst.ne[2] != 1 || request.dst.ne[3] != 1) {
    return false;
  }

  if (is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst)) {
    const float * a = static_cast<const float *>(request.src0.data);
    const float * b = static_cast<const float *>(request.src1.data);
    float * c = static_cast<float *>(request.dst.data);

    for (uint64_t i = 0; i < m; ++i) {
      for (uint64_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (uint64_t p = 0; p < k; ++p) {
          acc += a[i * k + p] * b[p * n + j];
        }
        c[i * n + j] = acc;
      }
    }
    return true;
  }

  for (uint64_t i = 0; i < m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < k; ++p) {
        acc += read_f32_at(request.src0, p, i) * read_f32_at(request.src1, j, p);
      }
      write_f32_at(request.dst, j, i, acc);
    }
  }

  return true;
}

template <class request_type>
inline bool run_soft_max(const request_type & request) noexcept {
  const uint64_t width = request.src0.ne[0];
  const uint64_t count = tensor_element_count(request.src0);
  if (width == 0 || count == 0 || count % width != 0 || count != tensor_element_count(request.dst)) {
    return false;
  }
  const uint64_t rows = count / width;

  if (is_dense_contiguous(request.src0) && is_dense_contiguous(request.dst)) {
    const float * src = static_cast<const float *>(request.src0.data);
    float * dst = static_cast<float *>(request.dst.data);
    for (uint64_t row = 0; row < rows; ++row) {
      const uint64_t offset = row * width;
      float max_v = src[offset];
      for (uint64_t i = 1; i < width; ++i) {
        max_v = std::max(max_v, src[offset + i]);
      }

      float sum = 0.0f;
      for (uint64_t i = 0; i < width; ++i) {
        const float e = std::exp(src[offset + i] - max_v);
        dst[offset + i] = e;
        sum += e;
      }

      for (uint64_t i = 0; i < width; ++i) {
        dst[offset + i] /= sum;
      }
    }
    return true;
  }

  for (uint64_t row = 0; row < rows; ++row) {
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

  return true;
}

template <class request_type>
inline bool run_unary_subop(const request_type & request) noexcept {
  const auto subop = static_cast<uint8_t>(request.subop);
  if (subop == 0) {
    return run_unary(request, [](const float v) { return std::fabs(v); });
  }
  if (subop == 2) {
    return run_unary(request, [](const float v) { return -v; });
  }
  if (subop == 6) {
    return run_unary(request, [](const float v) { return std::max(0.0f, v); });
  }
  if (subop == 13) {
    return run_unary(request, [](const float v) { return std::exp(v); });
  }
  return false;
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
  if (k == 0 || m == 0 || n == 0) {
    return false;
  }
  return request.src1.ne[1] == k && request.dst.ne[0] == n &&
         request.dst.ne[1] == m && request.src0.ne[2] == 1 &&
         request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
         request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
         request.dst.ne[3] == 1;
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
  if (subop != 0 && subop != 2 && subop != 6 && subop != 13) {
    return false;
  }
  return can_run_unary(request);
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
    return can_run_unary_subop(request);
  }
  return false;
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
  } else if constexpr (std::is_same_v<request_type, event::op_unary>) {
    (void) run_unary_subop(request);
  }
}

template <class request_type>
inline bool execute_scalar(const request_type & request) noexcept {
  if (!can_execute_scalar(request)) {
    return false;
  }
  execute_scalar_unchecked(request);
  return true;
}

}  // namespace emel::kernel::detail
