#pragma once

#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/events.hpp"

namespace emel::kernel::aarch64::guard {

namespace detail {

inline bool can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x8_request(
    const ::emel::kernel::event::op_mul_mat & request,
    const uint8_t packed_dtype) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q4_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  const size_t dst_row_bytes = sizeof(float) * m;
  return k != 0u &&
      m != 0u &&
      rhs_rows == ::emel::kernel::detail::quant::Q4_K_X8_ROWS &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == rhs_rows &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) == packed_dtype &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_k_x8 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      request.src0.nb[0] == 1u &&
      group_bytes != 0u &&
      request.src0.nb[1] == group_bytes &&
      request.src0.nb[2] == group_bytes * group_count &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      rhs_row_bytes != 0u &&
      request.src1.nb[1] == rhs_row_bytes &&
      request.src1.nb[2] == rhs_row_bytes * rhs_rows &&
      request.src1.nb[3] == request.src1.nb[2] &&
      request.dst.nb[0] == dst_row_bytes &&
      request.dst.nb[1] == sizeof(float) &&
      request.dst.nb[2] == dst_row_bytes * rhs_rows &&
      request.dst.nb[3] == request.dst.nb[2];
}

inline bool can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8_request(
    const ::emel::kernel::event::op_mul_mat & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  const size_t dst_row_bytes = sizeof(float) * m;
  return k != 0u &&
      m != 0u &&
      rhs_rows == ::emel::kernel::detail::quant::Q6_K_X8_ROWS &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == rhs_rows &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k_x8_q8_prepared &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_k_x8 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      request.src0.nb[0] == 1u &&
      group_bytes != 0u &&
      request.src0.nb[1] == group_bytes &&
      request.src0.nb[2] == group_bytes * group_count &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      rhs_row_bytes != 0u &&
      request.src1.nb[1] == rhs_row_bytes &&
      request.src1.nb[2] == rhs_row_bytes * rhs_rows &&
      request.src1.nb[3] == request.src1.nb[2] &&
      request.dst.nb[0] == dst_row_bytes &&
      request.dst.nb[1] == sizeof(float) &&
      request.dst.nb[2] == dst_row_bytes * rhs_rows &&
      request.dst.nb[3] == request.dst.nb[2];
}

inline bool can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8(
    const ::emel::kernel::event::op_mul_mat & request,
    const bool neon_available) noexcept {
  return neon_available &&
      ::emel::kernel::aarch64::detail::neon_q4_vector_packed_supported() &&
      can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x8_request(
          request, ::emel::kernel::detail::dtype_q4_k_x8_bl8);
}

inline bool can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8(
    const ::emel::kernel::event::op_mul_mat & request,
    const bool neon_available) noexcept {
  return neon_available &&
      ::emel::kernel::aarch64::detail::neon_q6_vector_prepared_q8_rhs_i8mm_supported() &&
      can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8_request(request);
}

}  // namespace detail

template <class dispatch_event_type>
struct simd_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon(ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q1_0_g128_vector_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q1_0_g128_vector_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl8_tail_safe(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8_full_groups {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl8_full_groups(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl8_matrix_x4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_packed(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_packed_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_prepared_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_argmax_q6_vector_packed_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_generic {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    return simd_op<::emel::kernel::aarch64::event::dispatch_op_mul_mat>{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl8_matrix_x4{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl8_full_groups{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl8{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl4{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_vector{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_prepared_q8_rhs{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_packed_q8_rhs{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_packed{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector{}(ev, ctx);
  }
};

struct simd_op_flash_attn_ext_f16kv_one_chunk {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_run_neon_flash_attn_ext_f16kv_one_chunk_request(
        ev.request, ctx.neon_available, ctx.flash_attn_workspace);
  }
};

struct valid_op_flash_attn_ext_shared {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_backend_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::detail::can_run_flash_attn_ext_with_workspace(
               ev.request, ctx.flash_attn_workspace) &&
        !simd_op_flash_attn_ext_f16kv_one_chunk{}(ev, ctx);
  }
};

template <class dispatch_event_type>
struct valid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_backend_request(ev.request)) {
      return false;
    }
    return !simd_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <class dispatch_event_type>
struct invalid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    return !simd_op<dispatch_event_type>{}(ev, ctx) &&
           !valid_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop>
struct unary_subop_is {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_unary & ev,
                  const action::context &) const noexcept {
    return ev.request.subop == subop;
  }
};

template <::emel::kernel::event::unary_subop subop>
using simd_op_unary_subop = ::emel::kernel::detail::simd_unary_subop_guard<
    ::emel::kernel::aarch64::event::dispatch_op_unary, action::context,
    simd_op<::emel::kernel::aarch64::event::dispatch_op_unary>, unary_subop_is<subop>>;

template <::emel::kernel::event::unary_subop subop>
using valid_op_unary_subop = ::emel::kernel::detail::valid_unary_subop_guard<
    ::emel::kernel::aarch64::event::dispatch_op_unary, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_unary>, unary_subop_is<subop>>;

using simd_op_unary_abs = simd_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using simd_op_unary_neg = simd_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using simd_op_unary_relu = simd_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_abs = valid_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using valid_op_unary_neg = valid_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using valid_op_unary_relu = valid_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_exp = valid_op_unary_subop<::emel::kernel::event::unary_subop::exp>;

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                                 \
  using simd_##op_name =                                                         \
      simd_op<::emel::kernel::aarch64::event::dispatch_##op_name>;               \
  using valid_##op_name =                                                        \
      valid_op<::emel::kernel::aarch64::event::dispatch_##op_name>;              \
  using invalid_##op_name =                                                      \
      invalid_op<::emel::kernel::aarch64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

}  // namespace emel::kernel::aarch64::guard
