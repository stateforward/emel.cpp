#pragma once

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/metal/actions.hpp"
#include "emel/kernel/metal/context.hpp"
#include "emel/kernel/metal/detail.hpp"
#include "emel/kernel/metal/errors.hpp"
#include "emel/kernel/metal/events.hpp"

// Metal actor guards. Every guard combines:
//   1. Metal readiness (device, library, pipelines, pool) in the context;
//   2. the shared dispatch-request validation and per-op shape/dtype/layout
//      contracts from src/emel/kernel/detail.hpp (the same predicates the
//      CPU backends use, so acceptance never drifts between backends);
//   3. staging-capacity bounds (each tensor must fit one pool slice).
// The guards are pure: they never run compute and never mutate context.
namespace emel::kernel::metal::guard {

namespace detail {

inline bool metal_ready(const action::context &ctx) noexcept {
  return ctx.metal_available && ctx.runtime != nullptr &&
         ctx.runtime->available();
}

inline bool
within_capacity(const ::emel::kernel::event::tensor_view &t) noexcept {
  return ::emel::kernel::metal::detail::tensor_storage_bytes(t) <=
         ::emel::kernel::metal::detail::k_pool_slice_capacity_bytes;
}

inline bool
within_capacity(const ::emel::kernel::event::tensor_view_mut &t) noexcept {
  return ::emel::kernel::metal::detail::tensor_storage_bytes(t) <=
         ::emel::kernel::metal::detail::k_pool_slice_capacity_bytes;
}

//------------------------------------------------------------------------------//
// Variant predicates (shape/dtype/layout contracts mirroring the shared
// can_run_* helpers; the destination dtype of a variant is enforced here so
// the actions never inspect dtype at runtime).
//------------------------------------------------------------------------------//

inline bool
can_mul_mat_f32(const ::emel::kernel::event::op_mul_mat &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool valid_shape = request.src1.ne[1] == k && request.dst.ne[0] == n &&
                           request.dst.ne[1] == m && request.src0.ne[2] == 1 &&
                           request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
                           request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
                           request.dst.ne[3] == 1;
  const bool f32_path = ::emel::kernel::detail::dtype_code(request.src0.type) ==
                            ::emel::kernel::detail::dtype_f32 &&
                        ::emel::kernel::detail::dtype_code(request.src1.type) ==
                            ::emel::kernel::detail::dtype_f32 &&
                        ::emel::kernel::detail::dtype_code(request.dst.type) ==
                            ::emel::kernel::detail::dtype_f32;
  return !has_empty_dim && valid_shape && f32_path;
}

inline bool
can_mul_mat_f16(const ::emel::kernel::event::op_mul_mat &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[1];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool valid_shape = request.src1.ne[0] == k && request.dst.ne[0] == m &&
                           request.dst.ne[1] == n && request.src0.ne[2] == 1 &&
                           request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
                           request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
                           request.dst.ne[3] == 1;
  const bool f16_path =
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_f16 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_f16 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::is_dense_contiguous(request.src0) &&
      ::emel::kernel::detail::is_dense_contiguous(request.src1) &&
      ::emel::kernel::detail::is_dense_contiguous(request.dst);
  return !has_empty_dim && valid_shape && f16_path;
}

inline bool
can_mul_mat_q8_0(const ::emel::kernel::event::op_mul_mat &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool valid_shape = request.src1.ne[1] == k && request.dst.ne[0] == n &&
                           request.dst.ne[1] == m && request.src0.ne[2] == 1 &&
                           request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
                           request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
                           request.dst.ne[3] == 1;
  const bool quantized_path =
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q8_0 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      (k % ::emel::kernel::detail::quant::QK8_0) == 0u &&
      (k / ::emel::kernel::detail::quant::QK8_0) <=
          ::emel::kernel::detail::quant::MAX_Q8_0_BLOCKS &&
      ::emel::kernel::detail::is_dense_contiguous(request.src1) &&
      ::emel::kernel::detail::is_dense_contiguous(request.dst) &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == ::emel::kernel::detail::quantized_row_storage_bytes(
                                ::emel::kernel::detail::dtype_q8_0, k) &&
      request.src0.nb[2] == request.src0.nb[1] * m &&
      request.src0.nb[3] == request.src0.nb[2];
  return !has_empty_dim && valid_shape && quantized_path;
}

inline bool
can_add_equal(const ::emel::kernel::event::op_add &request) noexcept {
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  return ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         count == ::emel::kernel::detail::tensor_element_count(request.src0) &&
         count == ::emel::kernel::detail::tensor_element_count(request.src1);
}

inline bool
can_add_broadcast_row(const ::emel::kernel::event::op_add &request) noexcept {
  const bool same_shape = request.src0.ne[0] == request.dst.ne[0] &&
                          request.src0.ne[1] == request.dst.ne[1] &&
                          request.src0.ne[2] == request.dst.ne[2] &&
                          request.src0.ne[3] == request.dst.ne[3];
  return same_shape && request.dst.ne[0] > 0 &&
         ::emel::kernel::detail::tensor_element_count(request.dst) >
             ::emel::kernel::detail::tensor_element_count(request.src1) &&
         request.src1.ne[0] == request.dst.ne[0] && request.src1.ne[1] == 1 &&
         request.src1.ne[2] == 1 && request.src1.ne[3] == 1 &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::has_valid_tensor_layout(request.src0) &&
         ::emel::kernel::detail::has_valid_tensor_layout(request.src1) &&
         ::emel::kernel::detail::has_valid_tensor_layout(request.dst);
}

// Supported subops mirror
// ::emel::kernel::detail::can_run_unary_subop (the shader implements the
// same set).
inline bool can_run_unary_subop_set(
    const ::emel::kernel::event::op_unary &request) noexcept {
  const auto subop = static_cast<uint8_t>(request.subop);
  return subop == ::emel::kernel::detail::unary_subop_abs ||
         subop == ::emel::kernel::detail::unary_subop_neg ||
         subop == ::emel::kernel::detail::unary_subop_tanh ||
         subop == ::emel::kernel::detail::unary_subop_elu ||
         subop == ::emel::kernel::detail::unary_subop_relu ||
         subop == ::emel::kernel::detail::unary_subop_gelu ||
         subop == ::emel::kernel::detail::unary_subop_silu ||
         subop == ::emel::kernel::detail::unary_subop_exp;
}

inline bool can_im2col_variant(const ::emel::kernel::event::op_im2col &request,
                               const bool f16_dst) noexcept {
  ::emel::kernel::detail::im2col_op_params params = {};
  if (!::emel::kernel::detail::read_im2col_params(request, params) ||
      params.is_2d != 0 || params.s0 <= 0 || params.d0 <= 0 || params.p0 < 0) {
    return false;
  }
  const int64_t kernel = static_cast<int64_t>(request.src0.ne[0]);
  const int64_t channels = static_cast<int64_t>(request.src0.ne[1]);
  const int64_t length = static_cast<int64_t>(request.src1.ne[0]);
  const int64_t out_length = ::emel::kernel::detail::conv_output_length(
      length, kernel, params.s0, params.p0, params.d0);
  const uint8_t src0_type =
      ::emel::kernel::detail::dtype_code(request.src0.type);
  const uint8_t dst_type = ::emel::kernel::detail::dtype_code(request.dst.type);
  const uint8_t expected_dst = f16_dst ? ::emel::kernel::detail::dtype_f16
                                       : ::emel::kernel::detail::dtype_f32;
  return request.src1.data != nullptr && request.dst.data != nullptr &&
         kernel > 0 && channels > 0 && length > 0 && out_length > 0 &&
         (src0_type == ::emel::kernel::detail::dtype_f32 ||
          src0_type == ::emel::kernel::detail::dtype_f16) &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         dst_type == expected_dst &&
         request.src1.ne[1] == static_cast<uint64_t>(channels) &&
         request.src1.ne[3] == 1 &&
         request.dst.ne[0] == static_cast<uint64_t>(channels * kernel) &&
         request.dst.ne[1] == static_cast<uint64_t>(out_length) &&
         request.dst.ne[2] == request.src1.ne[2] && request.dst.ne[3] == 1 &&
         ::emel::kernel::detail::has_valid_tensor_layout(request.src1) &&
         ::emel::kernel::detail::is_dense_contiguous(request.dst);
}

inline bool can_conv_transpose_1d_variant(
    const ::emel::kernel::event::op_conv_transpose_1d &request,
    const bool f16_weights) noexcept {
  int32_t s0 = 0;
  int32_t p0 = 0;
  int32_t d0 = 0;
  const uint8_t src0_type =
      ::emel::kernel::detail::dtype_code(request.src0.type);
  if (!::emel::kernel::detail::read_op_param_i32(
          request.op_params.data(), request.op_params_size, 0u, s0) ||
      !::emel::kernel::detail::read_op_param_i32(
          request.op_params.data(), request.op_params_size, 1u, p0) ||
      !::emel::kernel::detail::read_op_param_i32(
          request.op_params.data(), request.op_params_size, 2u, d0) ||
      s0 <= 0 || p0 != 0 || d0 != 1) {
    return false;
  }
  const int64_t kernel = static_cast<int64_t>(request.src0.ne[0]);
  const int64_t out_channels = static_cast<int64_t>(request.src0.ne[1]);
  const int64_t in_channels = static_cast<int64_t>(request.src0.ne[2]);
  const int64_t length = static_cast<int64_t>(request.src1.ne[0]);
  const int64_t out_length = (length - 1) * s0 + kernel;
  const uint8_t expected_src0 = f16_weights ? ::emel::kernel::detail::dtype_f16
                                            : ::emel::kernel::detail::dtype_f32;
  return request.src0.data != nullptr && request.src1.data != nullptr &&
         request.dst.data != nullptr && kernel > 0 && out_channels > 0 &&
         in_channels > 0 && length > 0 && out_length > 0 &&
         src0_type == expected_src0 &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         request.src0.ne[3] == 1 &&
         request.src1.ne[1] == static_cast<uint64_t>(in_channels) &&
         request.src1.ne[2] == 1 && request.src1.ne[3] == 1 &&
         request.dst.ne[0] == static_cast<uint64_t>(out_length) &&
         request.dst.ne[1] == static_cast<uint64_t>(out_channels) &&
         request.dst.ne[2] == 1 && request.dst.ne[3] == 1 &&
         ::emel::kernel::detail::has_valid_tensor_layout(request.src0) &&
         ::emel::kernel::detail::has_valid_tensor_layout(request.src1) &&
         ::emel::kernel::detail::is_dense_contiguous(request.dst);
}

inline bool
can_get_rows_variant(const ::emel::kernel::event::op_get_rows &request,
                     const bool f16_src0) noexcept {
  const uint8_t src0_type =
      ::emel::kernel::detail::dtype_code(request.src0.type);
  const uint8_t src1_type =
      ::emel::kernel::detail::dtype_code(request.src1.type);
  const uint8_t dst_type = ::emel::kernel::detail::dtype_code(request.dst.type);
  const uint64_t cols = request.src0.ne[0];
  const uint64_t rows = request.src0.ne[1];
  const bool shapes_ok = cols > 0 && rows > 0 && request.dst.ne[0] == cols &&
                         request.dst.ne[1] == request.src1.ne[0] &&
                         request.dst.ne[2] == request.src1.ne[1] &&
                         request.dst.ne[3] == request.src1.ne[2] &&
                         request.src0.ne[2] == request.src1.ne[1] &&
                         request.src0.ne[3] == request.src1.ne[2] &&
                         request.src1.ne[3] == 1;
  const uint8_t expected_src0 = f16_src0 ? ::emel::kernel::detail::dtype_f16
                                         : ::emel::kernel::detail::dtype_f32;
  const bool types_ok = src0_type == expected_src0 &&
                        src1_type == ::emel::kernel::detail::dtype_i32 &&
                        dst_type == ::emel::kernel::detail::dtype_f32;
  const bool layouts_ok =
      ::emel::kernel::detail::has_valid_tensor_layout(request.src0) &&
      ::emel::kernel::detail::has_valid_tensor_layout(request.src1) &&
      ::emel::kernel::detail::is_dense_contiguous(request.dst);
  return shapes_ok && types_ok && layouts_ok;
}

//------------------------------------------------------------------------------//
// Per-op acceptance combiners.
//------------------------------------------------------------------------------//

inline bool
valid_mul_mat(const ::emel::kernel::event::op_mul_mat &request) noexcept {
  return ::emel::kernel::detail::validate_dispatch_request(request) &&
         (can_mul_mat_f32(request) || can_mul_mat_f16(request) ||
          can_mul_mat_q8_0(request)) &&
         within_capacity(request.src0) && within_capacity(request.src1) &&
         within_capacity(request.dst);
}

inline bool valid_add(const ::emel::kernel::event::op_add &request) noexcept {
  return ::emel::kernel::detail::validate_dispatch_request(request) &&
         (can_add_equal(request) || can_add_broadcast_row(request)) &&
         within_capacity(request.src0) && within_capacity(request.src1) &&
         within_capacity(request.dst);
}

inline bool
valid_unary(const ::emel::kernel::event::op_unary &request) noexcept {
  return ::emel::kernel::detail::validate_dispatch_request(request) &&
         can_run_unary_subop_set(request) &&
         ::emel::kernel::detail::can_run_unary(request) &&
         within_capacity(request.src0) && within_capacity(request.dst);
}

inline bool
valid_im2col(const ::emel::kernel::event::op_im2col &request) noexcept {
  return ::emel::kernel::detail::validate_dispatch_request(request) &&
         (can_im2col_variant(request, false) ||
          can_im2col_variant(request, true)) &&
         within_capacity(request.src0) && within_capacity(request.src1) &&
         within_capacity(request.dst);
}

inline bool valid_conv_transpose_1d(
    const ::emel::kernel::event::op_conv_transpose_1d &request) noexcept {
  return ::emel::kernel::detail::validate_dispatch_request(request) &&
         (can_conv_transpose_1d_variant(request, false) ||
          can_conv_transpose_1d_variant(request, true)) &&
         within_capacity(request.src0) && within_capacity(request.src1) &&
         within_capacity(request.dst);
}

inline bool
valid_get_rows(const ::emel::kernel::event::op_get_rows &request) noexcept {
  return ::emel::kernel::detail::validate_dispatch_request(request) &&
         (can_get_rows_variant(request, false) ||
          can_get_rows_variant(request, true)) &&
         within_capacity(request.src0) && within_capacity(request.src1) &&
         within_capacity(request.dst);
}

} // namespace detail

//------------------------------------------------------------------------------//
// Transition guards.
//------------------------------------------------------------------------------//

struct valid_op_mul_mat_f32 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) && detail::can_mul_mat_f32(ev.request) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct valid_op_mul_mat_f16 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) && detail::can_mul_mat_f16(ev.request) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct valid_op_mul_mat_q8_0 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) && detail::can_mul_mat_q8_0(ev.request) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct invalid_op_mul_mat {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  const action::context &) const noexcept {
    return !detail::valid_mul_mat(ev.request);
  }
};

struct valid_op_add_equal {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_add &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) && detail::can_add_equal(ev.request) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct valid_op_add_broadcast_row {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_add &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_add_broadcast_row(ev.request) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct invalid_op_add {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_add &ev,
                  const action::context &) const noexcept {
    return !detail::valid_add(ev.request);
  }
};

template <::emel::kernel::event::unary_subop subop>
struct valid_op_unary_subop {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_unary &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request) &&
           ::emel::kernel::detail::can_run_unary(ev.request) &&
           ev.request.subop == subop &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.dst);
  }
};

struct invalid_op_unary {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_unary &ev,
                  const action::context &) const noexcept {
    return !detail::valid_unary(ev.request);
  }
};

struct valid_op_im2col_f32 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_im2col &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_im2col_variant(ev.request, false) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct valid_op_im2col_f16 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_im2col &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_im2col_variant(ev.request, true) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct invalid_op_im2col {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_im2col &ev,
                  const action::context &) const noexcept {
    return !detail::valid_im2col(ev.request);
  }
};

struct valid_op_conv_transpose_1d_f32 {
  bool operator()(
      const ::emel::kernel::metal::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_conv_transpose_1d_variant(ev.request, false) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct valid_op_conv_transpose_1d_f16 {
  bool operator()(
      const ::emel::kernel::metal::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_conv_transpose_1d_variant(ev.request, true) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct invalid_op_conv_transpose_1d {
  bool operator()(
      const ::emel::kernel::metal::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &) const noexcept {
    return !detail::valid_conv_transpose_1d(ev.request);
  }
};

struct valid_op_get_rows_f32 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_get_rows &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_get_rows_variant(ev.request, false) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct valid_op_get_rows_f16 {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_get_rows &ev,
                  const action::context &ctx) const noexcept {
    return detail::metal_ready(ctx) &&
           detail::can_get_rows_variant(ev.request, true) &&
           detail::within_capacity(ev.request.src0) &&
           detail::within_capacity(ev.request.src1) &&
           detail::within_capacity(ev.request.dst) &&
           ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

struct invalid_op_get_rows {
  bool operator()(const ::emel::kernel::metal::event::dispatch_op_get_rows &ev,
                  const action::context &) const noexcept {
    return !detail::valid_get_rows(ev.request);
  }
};

} // namespace emel::kernel::metal::guard
