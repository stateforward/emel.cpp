#pragma once

#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/events.hpp"
#include "emel/kernel/detail.hpp"

namespace emel::kernel::aarch64::guard {

namespace detail {

// Routing predicates for the multi-RHS matrix_x4 packed matmul rows: these
// decide transition acceptance, so they live with the guards (the x8
// variants above follow the same placement).
// Batch-major f32 dst contract shared by the multi-RHS packed matmul kernels.
// nb[0] is the column stride between per-RHS result rows and nb[1] the dense
// element stride. Accepts the dense full-output view (nb[2] spans all RHS
// rows) and the group-aligned row-slice lane view emitted by the parallel
// matmul slicer, whose slice is embedded in a wider output (nb[0] covers the
// full output row while ne[1]/nb[2] describe only the slice rows).
inline bool is_batch_major_dst_view(const ::emel::kernel::event::tensor_view_mut &dst,
                                    const uint64_t m,
                                    const uint64_t rhs_rows) noexcept {
  // Keep every byte-stride term in uint64_t: mixing size_t in the products
  // could truncate on 32-bit hosts and mis-accept a stride contract.
  constexpr uint64_t k_f32_bytes = sizeof(float);
  const uint64_t slice_row_bytes = k_f32_bytes * m;
  const bool full_view = dst.nb[2] == dst.nb[0] * rhs_rows;
  const bool row_slice_view = dst.nb[2] == slice_row_bytes;
  return dst.nb[1] == k_f32_bytes && (dst.nb[0] % k_f32_bytes) == 0u &&
         dst.nb[0] >= slice_row_bytes && dst.nb[3] == dst.nb[2] &&
         (full_view || row_slice_view);
}

inline bool can_use_neon_mul_mat_q8_0_packed_bl8_matrix_x4(
    const ::emel::kernel::event::op_mul_mat &request, const bool neon_available) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t lhs_group_count =
      ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(m);
  const uint64_t rhs_group_count =
      ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(rhs_rows);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(k);
  return neon_available && ::emel::kernel::aarch64::detail::neon_q8_0_packed_bl8_supported() && k != 0u &&
         m != 0u && rhs_rows == ::emel::kernel::detail::quant::Q8_0_X4_ROWS &&
         request.src1.ne[1] == k && request.dst.ne[0] == rhs_rows &&
         request.dst.ne[1] == m && request.src0.ne[2] == 1u &&
         request.src0.ne[3] == 1u && request.src1.ne[2] == 1u &&
         request.src1.ne[3] == 1u && request.dst.ne[2] == 1u &&
         request.dst.ne[3] == 1u &&
         (m % ::emel::kernel::detail::quant::Q8_0_X4_ROWS) == 0u &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_q8_0_x4_bl8 &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_q8_0_x4_bl8 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         group_bytes != 0u && request.src0.nb[0] == 1u &&
         request.src0.nb[1] == group_bytes &&
         request.src0.nb[2] == group_bytes * lhs_group_count &&
         request.src0.nb[3] == request.src0.nb[2] && request.src1.nb[0] == 1u &&
         request.src1.nb[1] == group_bytes &&
         request.src1.nb[2] == group_bytes * rhs_group_count &&
         request.src1.nb[3] == request.src1.nb[2] &&
         is_batch_major_dst_view(request.dst, m, rhs_rows);
}

inline bool can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x4_request(
    const ::emel::kernel::event::op_mul_mat &request, const uint8_t packed_dtype) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t group_count =
      ::emel::kernel::detail::quant::packed_q4_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return k != 0u && m != 0u &&
         rhs_rows == ::emel::kernel::detail::quant::Q8_0_X4_ROWS &&
         request.src1.ne[1] == k && request.dst.ne[0] == rhs_rows &&
         request.dst.ne[1] == m && request.src0.ne[2] == 1u &&
         request.src0.ne[3] == 1u && request.src1.ne[2] == 1u &&
         request.src1.ne[3] == 1u && request.dst.ne[2] == 1u &&
         request.dst.ne[3] == 1u &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             packed_dtype &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_q8_k_x4 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         request.src0.nb[0] == 1u && group_bytes != 0u &&
         request.src0.nb[1] == group_bytes &&
         request.src0.nb[2] == group_bytes * group_count &&
         request.src0.nb[3] == request.src0.nb[2] && request.src1.nb[0] == 1u &&
         rhs_row_bytes != 0u && request.src1.nb[1] == rhs_row_bytes &&
         request.src1.nb[2] == rhs_row_bytes * rhs_rows &&
         request.src1.nb[3] == request.src1.nb[2] &&
         is_batch_major_dst_view(request.dst, m, rhs_rows);
}

inline bool can_run_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4_request(
    const ::emel::kernel::event::op_mul_mat &request, const uint8_t packed_dtype,
    const size_t group_bytes) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t group_count =
      ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return k != 0u && m != 0u &&
         rhs_rows == ::emel::kernel::detail::quant::Q8_0_X4_ROWS &&
         request.src1.ne[1] == k && request.dst.ne[0] == rhs_rows &&
         request.dst.ne[1] == m && request.src0.ne[2] == 1u &&
         request.src0.ne[3] == 1u && request.src1.ne[2] == 1u &&
         request.src1.ne[3] == 1u && request.dst.ne[2] == 1u &&
         request.dst.ne[3] == 1u &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             packed_dtype &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_q8_k_x4 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         request.src0.nb[0] == 1u && group_bytes != 0u &&
         request.src0.nb[1] == group_bytes &&
         request.src0.nb[2] == group_bytes * group_count &&
         request.src0.nb[3] == request.src0.nb[2] && request.src1.nb[0] == 1u &&
         rhs_row_bytes != 0u && request.src1.nb[1] == rhs_row_bytes &&
         request.src1.nb[2] == rhs_row_bytes * rhs_rows &&
         request.src1.nb[3] == request.src1.nb[2] &&
         is_batch_major_dst_view(request.dst, m, rhs_rows);
}

inline bool can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4(
    const ::emel::kernel::event::op_mul_mat &request, const bool neon_available) noexcept {
  return neon_available && ::emel::kernel::aarch64::detail::neon_q4_vector_packed_supported() &&
         can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x4_request(
             request, ::emel::kernel::detail::dtype_q4_k_x8_bl4);
}

inline bool can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4(
    const ::emel::kernel::event::op_mul_mat &request, const bool neon_available) noexcept {
  return neon_available && ::emel::kernel::aarch64::detail::neon_q4_vector_packed_supported() &&
         can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x4_request(
             request, ::emel::kernel::detail::dtype_q4_k_x8_bl8);
}

inline bool can_use_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4(
    const ::emel::kernel::event::op_mul_mat &request, const bool neon_available) noexcept {
  return neon_available && ::emel::kernel::aarch64::detail::neon_q6_vector_packed_supported() &&
         can_run_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4_request(
             request, ::emel::kernel::detail::dtype_q6_k_x8,
             ::emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(
                 request.src0.ne[0]));
}

inline bool can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4(
    const ::emel::kernel::event::op_mul_mat &request, const bool neon_available) noexcept {
  return neon_available && ::emel::kernel::aarch64::detail::neon_q6_vector_prepared_q8_rhs_i8mm_supported() &&
         can_run_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4_request(
             request, ::emel::kernel::detail::dtype_q6_k_x8_q8_prepared,
             ::emel::kernel::detail::quant::
                 prepared_q6_k_x8_q8_group_storage_bytes(request.src0.ne[0]));
}


inline bool can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x8_request(
    const ::emel::kernel::event::op_mul_mat &request,
    const uint8_t packed_dtype) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t group_count =
      ::emel::kernel::detail::quant::packed_q4_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
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
      is_batch_major_dst_view(request.dst, m, rhs_rows);
}

inline bool
can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8_request(
    const ::emel::kernel::event::op_mul_mat &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t group_count =
      ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
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
      is_batch_major_dst_view(request.dst, m, rhs_rows);
}

inline bool can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8(
    const ::emel::kernel::event::op_mul_mat &request,
    const bool neon_available) noexcept {
  return neon_available &&
         ::emel::kernel::aarch64::detail::neon_q4_vector_packed_supported() &&
         can_run_neon_mul_mat_q4_vector_packed_q8_rhs_matrix_x8_request(
             request, ::emel::kernel::detail::dtype_q4_k_x8_bl8);
}

inline bool can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8(
    const ::emel::kernel::event::op_mul_mat &request,
    const bool neon_available) noexcept {
  return neon_available &&
         ::emel::kernel::aarch64::detail::
             neon_q6_vector_prepared_q8_rhs_i8mm_supported() &&
         can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8_request(
             request);
}

// f16 x f16 mul_mat NEON variant: same request contract as the shared
// ggml-layout f16 path (can_run_mul_mat_f16); additionally requires fp16
// vector arithmetic because the kernel accumulates in fp16 lanes, matching
// what the reference computes on fp16-capable aarch64 hosts.
inline bool can_use_neon_mul_mat_f16_vector(
    const ::emel::kernel::event::op_mul_mat &request,
    const bool neon_available) noexcept {
  return neon_available &&
         ::emel::kernel::aarch64::detail::neon_f16_vector_supported() &&
         ::emel::kernel::detail::can_run_mul_mat_f16(request);
}

inline bool can_use_neon_conv_transpose_1d_f32(
    const ::emel::kernel::event::op_conv_transpose_1d &request,
    const bool neon_available) noexcept {
  return neon_available &&
         ::emel::kernel::aarch64::detail::neon_conv_transpose_f32_supported() &&
         ::emel::kernel::detail::can_run_conv_transpose_1d(request) &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::tensor_stride_bytes(request.src0, 0) ==
             sizeof(float);
}

} // namespace detail

template <class dispatch_event_type> struct simd_op {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon(ev.request,
                                                         ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q5_0_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q5_0_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_0_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q4_0_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_1_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q4_1_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_f16_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::can_use_neon_mul_mat_f16_vector(ev.request,
                                                   ctx.neon_available);
  }
};

struct simd_op_conv_transpose_1d_f32 {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::can_use_neon_conv_transpose_1d_f32(ev.request,
                                                      ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q8_0_packed_bl8_tail_safe(ev.request,
                                                       ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8_full_groups {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q8_0_packed_bl8_full_groups(ev.request,
                                                         ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::can_use_neon_mul_mat_q8_0_packed_bl8_matrix_x4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q8_0_packed_bl4(ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4(ev.request,
                                                         ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::
        can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8(ev.request,
                                                         ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::
        can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8_matrix_x8(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q6_vector_packed(ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q6_vector_packed_q8_rhs(ev.request,
                                                     ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::
        can_use_neon_mul_mat_q6_vector_packed_q8_rhs_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q6_vector_prepared_q8_rhs(ev.request,
                                                       ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm(ev.request,
                                                            ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::
        can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return detail::
        can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_packed_q8_rhs(ev.request,
                                                            ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_generic {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    return simd_op<::emel::kernel::aarch64::event::dispatch_op_mul_mat>{}(
               ev, ctx) &&
           !simd_op_mul_mat_q5_0_vector{}(ev, ctx) &&
           !simd_op_mul_mat_q4_0_vector{}(ev, ctx) &&
           !simd_op_mul_mat_q4_1_vector{}(ev, ctx) &&
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
           !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x8{}(ev,
                                                                       ctx) &&
           !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_matrix_x4{}(ev,
                                                                       ctx) &&
           !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm{}(ev, ctx) &&
           !simd_op_mul_mat_q6_vector_prepared_q8_rhs{}(ev, ctx) &&
           !simd_op_mul_mat_q6_vector_packed_q8_rhs_matrix_x4{}(ev, ctx) &&
           !simd_op_mul_mat_q6_vector_packed_q8_rhs{}(ev, ctx) &&
           !simd_op_mul_mat_q6_vector_packed{}(ev, ctx) &&
           !simd_op_mul_mat_q6_vector{}(ev, ctx);
  }
};

struct simd_op_flash_attn_ext_f16kv_one_chunk {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_run_neon_flash_attn_ext_f16kv_one_chunk_request(
            ev.request, ctx.neon_available, ctx.flash_attn_workspace);
  }
};

struct valid_op_flash_attn_ext_shared {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext &ev,
      const action::context &ctx) const noexcept {
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

template <class dispatch_event_type> struct valid_op {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_backend_request(ev.request)) {
      return false;
    }
    return !simd_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <class dispatch_event_type> struct invalid_op {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &ctx) const noexcept {
    return !simd_op<dispatch_event_type>{}(ev, ctx) &&
           !valid_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop> struct unary_subop_is {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_unary &ev,
                  const action::context &) const noexcept {
    return ev.request.subop == subop;
  }
};

template <::emel::kernel::event::unary_subop subop>
using simd_op_unary_subop = ::emel::kernel::detail::simd_unary_subop_guard<
    ::emel::kernel::aarch64::event::dispatch_op_unary, action::context,
    simd_op<::emel::kernel::aarch64::event::dispatch_op_unary>,
    unary_subop_is<subop>>;

template <::emel::kernel::event::unary_subop subop>
using valid_op_unary_subop = ::emel::kernel::detail::valid_unary_subop_guard<
    ::emel::kernel::aarch64::event::dispatch_op_unary, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_unary>,
    unary_subop_is<subop>>;

using simd_op_unary_abs =
    simd_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using simd_op_unary_neg =
    simd_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using simd_op_unary_relu =
    simd_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_abs =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using valid_op_unary_neg =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using valid_op_unary_relu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_exp =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::exp>;
using valid_op_unary_tanh =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::tanh>;
using valid_op_unary_elu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::elu>;
using valid_op_unary_gelu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::gelu>;
using valid_op_unary_silu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::silu>;

// Variant predicates for the ops whose dtype/mode choice is modeled as
// explicit transition rows (op_unary pattern).
template <uint8_t src_dtype_code> struct get_rows_src_dtype_is {
  bool
  operator()(const ::emel::kernel::aarch64::event::dispatch_op_get_rows &ev,
             const action::context &) const noexcept {
    return ::emel::kernel::detail::dtype_code(ev.request.src0.type) ==
           src_dtype_code;
  }
};

template <bool neox_mode> struct rope_mode_is {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_rope &ev,
                  const action::context &) const noexcept {
    int32_t mode = 0;
    return ::emel::kernel::detail::read_op_param_i32(
               ev.request.op_params.data(), ev.request.op_params_size, 2u,
               mode) &&
           mode == (neox_mode ? ::emel::kernel::detail::rope_mode_neox
                              : ::emel::kernel::detail::rope_mode_norm);
  }
};

template <bool f16_dst> struct im2col_dst_dtype_is {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_im2col &ev,
                  const action::context &) const noexcept {
    return ::emel::kernel::detail::dtype_code(ev.request.dst.type) ==
           (f16_dst ? ::emel::kernel::detail::dtype_f16
                    : ::emel::kernel::detail::dtype_f32);
  }
};

template <bool f16_weights> struct conv_transpose_1d_weight_dtype_is {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &) const noexcept {
    return ::emel::kernel::detail::dtype_code(ev.request.src0.type) ==
           (f16_weights ? ::emel::kernel::detail::dtype_f16
                        : ::emel::kernel::detail::dtype_f32);
  }
};

template <uint8_t src_dtype_code>
using valid_op_get_rows_src = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_get_rows, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_get_rows>,
    get_rows_src_dtype_is<src_dtype_code>>;

using valid_op_get_rows_f32 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_f32>;
using valid_op_get_rows_f16 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_f16>;
using valid_op_get_rows_bf16 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_bf16>;
using valid_op_get_rows_q4_0 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_q4_0>;
using valid_op_get_rows_q8_0 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_q8_0>;
using valid_op_get_rows_q4_k =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_q4_k>;

template <class dispatch_event_type> struct mul_mat_f16_is {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &) const noexcept {
    return ::emel::kernel::detail::can_run_mul_mat_f16(ev.request);
  }
};

using valid_op_mul_mat_f16_base = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_mul_mat, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_mul_mat>,
    mul_mat_f16_is<::emel::kernel::aarch64::event::dispatch_op_mul_mat>>;

// Shared-path f16 row guard: mutually exclusive with the NEON f16 variant so
// the f16 route is decided entirely by guards (capability + request shape),
// never by transition-row ordering.
struct valid_op_mul_mat_f16 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    return valid_op_mul_mat_f16_base{}(ev, ctx) &&
           !simd_op_mul_mat_f16_vector{}(ev, ctx);
  }
};

using valid_op_rope_norm = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_rope, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_rope>,
    rope_mode_is<false>>;
using valid_op_rope_neox = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_rope, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_rope>,
    rope_mode_is<true>>;

using valid_op_im2col_f32 = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_im2col, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_im2col>,
    im2col_dst_dtype_is<false>>;
using valid_op_im2col_f16 = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_im2col, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_im2col>,
    im2col_dst_dtype_is<true>>;

using valid_op_conv_transpose_1d_f32_base =
    ::emel::kernel::detail::valid_variant_guard<
        ::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d,
        action::context,
        valid_op<::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d>,
        conv_transpose_1d_weight_dtype_is<false>>;

// Shared-path f32 row guard: mutually exclusive with the NEON f32 variant so
// the conv_transpose route is decided entirely by guards, never by
// transition-row ordering.
struct valid_op_conv_transpose_1d_f32 {
  bool operator()(
      const ::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &ctx) const noexcept {
    return valid_op_conv_transpose_1d_f32_base{}(ev, ctx) &&
           !simd_op_conv_transpose_1d_f32{}(ev, ctx);
  }
};
using valid_op_conv_transpose_1d_f16 =
    ::emel::kernel::detail::valid_variant_guard<
        ::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d,
        action::context,
        valid_op<::emel::kernel::aarch64::event::dispatch_op_conv_transpose_1d>,
        conv_transpose_1d_weight_dtype_is<true>>;

template <class dispatch_event_type> struct binary_equal_shape_is {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &) const noexcept {
    return ::emel::kernel::detail::can_run_binary(ev.request);
  }
};

template <class dispatch_event_type> struct binary_broadcast_row_is {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &) const noexcept {
    return ::emel::kernel::detail::can_run_binary_broadcast_row(ev.request);
  }
};

using valid_op_add_equal = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_add, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_add>,
    binary_equal_shape_is<::emel::kernel::aarch64::event::dispatch_op_add>>;
using valid_op_add_broadcast_row = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_add, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_add>,
    binary_broadcast_row_is<::emel::kernel::aarch64::event::dispatch_op_add>>;
using valid_op_mul_equal = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_mul, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_mul>,
    binary_equal_shape_is<::emel::kernel::aarch64::event::dispatch_op_mul>>;
using valid_op_mul_broadcast_row = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::aarch64::event::dispatch_op_mul, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_mul>,
    binary_broadcast_row_is<::emel::kernel::aarch64::event::dispatch_op_mul>>;

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                               \
  using simd_##op_name =                                                       \
      simd_op<::emel::kernel::aarch64::event::dispatch_##op_name>;             \
  using valid_##op_name =                                                      \
      valid_op<::emel::kernel::aarch64::event::dispatch_##op_name>;            \
  using invalid_##op_name =                                                    \
      invalid_op<::emel::kernel::aarch64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

} // namespace emel::kernel::aarch64::guard
