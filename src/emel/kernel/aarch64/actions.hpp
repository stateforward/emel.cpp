#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "emel/emel.h"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/errors.hpp"
#include "emel/kernel/aarch64/events.hpp"

namespace emel::kernel::aarch64::detail {

namespace event = ::emel::kernel::event;

inline bool detect_neon() noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  return true;
#else
  return false;
#endif
}

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type & tensor) noexcept {
  return ::emel::kernel::detail::is_dense_contiguous(tensor);
}

template <class request_type>
inline constexpr bool simd_supported_request_v =
    std::is_same_v<request_type, event::op_dup> ||
    std::is_same_v<request_type, event::op_add> ||
    std::is_same_v<request_type, event::op_sub> ||
    std::is_same_v<request_type, event::op_mul> ||
    std::is_same_v<request_type, event::op_div> ||
    std::is_same_v<request_type, event::op_sqr> ||
    std::is_same_v<request_type, event::op_sqrt> ||
    std::is_same_v<request_type, event::op_mul_mat> ||
    std::is_same_v<request_type, event::op_unary>;

inline bool unary_subop_supported_simd(const event::unary_subop subop) noexcept {
  const auto subop_code = static_cast<uint8_t>(subop);
  return subop_code == static_cast<uint8_t>(event::unary_subop::abs) ||
         subop_code == static_cast<uint8_t>(event::unary_subop::neg) ||
         subop_code == static_cast<uint8_t>(event::unary_subop::relu);
}

inline void execute_neon_unary_abs(const float * src, float * dst, const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    vst1q_f32(dst + i, vabsq_f32(v));
  }
  for (; i < count; ++i) {
    dst[i] = std::fabs(src[i]);
  }
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
}

inline void execute_neon_unary_neg(const float * src, float * dst, const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    vst1q_f32(dst + i, vnegq_f32(v));
  }
  for (; i < count; ++i) {
    dst[i] = -src[i];
  }
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
}

inline void execute_neon_unary_relu(const float * src, float * dst, const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  uint64_t i = 0;
  const float32x4_t zero = vdupq_n_f32(0.0f);
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    vst1q_f32(dst + i, vmaxq_f32(v, zero));
  }
  for (; i < count; ++i) {
    dst[i] = std::max(0.0f, src[i]);
  }
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
}

template <class request_type>
inline bool can_use_neon(const request_type & request, const bool neon_available) noexcept {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  (void) request;
  (void) neon_available;
  return false;
#else
  if constexpr (!simd_supported_request_v<request_type>) {
    return false;
  }

  const uint8_t src0_type = ::emel::kernel::detail::dtype_code(request.src0.type);
  const uint8_t dst_type = ::emel::kernel::detail::dtype_code(request.dst.type);
  const bool quantized_mul_mat =
      std::is_same_v<request_type, event::op_mul_mat> &&
      ::emel::kernel::detail::is_quantized_k_dtype(src0_type);
  const bool base_supported = neon_available &&
      ::emel::kernel::detail::can_run_backend_request(request) &&
      dst_type == ::emel::kernel::detail::dtype_f32 &&
      (quantized_mul_mat || src0_type == ::emel::kernel::detail::dtype_f32);

  bool src1_supported = true;
  if constexpr (::emel::kernel::detail::requires_src1_v<request_type>) {
    src1_supported =
        ::emel::kernel::detail::dtype_code(request.src1.type) ==
            ::emel::kernel::detail::dtype_f32 &&
        is_dense_contiguous(request.src1);
  }

  bool unary_supported = true;
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    unary_supported = unary_subop_supported_simd(request.subop);
  }

  return base_supported &&
      src1_supported &&
      unary_supported &&
      (quantized_mul_mat || is_dense_contiguous(request.src0)) &&
      is_dense_contiguous(request.dst);
#endif
}

inline bool can_run_neon_mul_mat_q6_vector_request(
    const event::op_mul_mat & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const size_t row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q6_k, k);
  return k != 0u &&
      m != 0u &&
      block_count != 0u &&
      block_count <= ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      request.src0.nb[0] == 1u &&
      row_bytes != 0u &&
      request.src0.nb[1] == row_bytes &&
      request.src0.nb[2] == row_bytes * m &&
      request.src0.nb[3] == request.src0.nb[2] &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
}

inline bool can_use_neon_mul_mat_q6_vector(const event::op_mul_mat & request,
                                           const bool neon_available) noexcept {
  return neon_available &&
      can_run_neon_mul_mat_q6_vector_request(request);
}

inline bool can_run_neon_mul_mat_q8_0_vector_request(
    const event::op_mul_mat & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  const size_t row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_0, k);
  return k != 0u &&
      m != 0u &&
      block_count != 0u &&
      block_count <= ::emel::kernel::detail::quant::MAX_Q8_0_BLOCKS &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q8_0 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      request.src0.nb[0] == 1u &&
      row_bytes != 0u &&
      request.src0.nb[1] == row_bytes &&
      request.src0.nb[2] == row_bytes * m &&
      request.src0.nb[3] == request.src0.nb[2] &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
}

inline bool can_use_neon_mul_mat_q8_0_vector(const event::op_mul_mat & request,
                                             const bool neon_available) noexcept {
  return neon_available &&
      can_run_neon_mul_mat_q8_0_vector_request(request);
}

inline bool neon_q8_0_packed_bl4_supported() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return true;
#else
  return false;
#endif
}

inline bool neon_q8_0_packed_bl8_supported() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  return true;
#else
  return false;
#endif
}

inline bool can_run_neon_mul_mat_q8_0_packed_request(const event::op_mul_mat & request,
                                                     const uint8_t packed_dtype) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_0, k);
  return k != 0u &&
      m != 0u &&
      block_count != 0u &&
      block_count <= ::emel::kernel::detail::quant::MAX_Q8_0_BLOCKS &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) == packed_dtype &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_0 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      request.src0.nb[0] == 1u &&
      group_bytes != 0u &&
      rhs_row_bytes != 0u &&
      request.src0.nb[1] == group_bytes &&
      request.src0.nb[2] == group_bytes * group_count &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      request.src1.nb[1] == rhs_row_bytes &&
      request.src1.nb[2] == rhs_row_bytes &&
      request.src1.nb[3] == rhs_row_bytes &&
      is_dense_contiguous(request.dst);
}

inline bool can_use_neon_mul_mat_q8_0_packed_bl4(const event::op_mul_mat & request,
                                                 const bool neon_available) noexcept {
  return neon_available &&
      neon_q8_0_packed_bl4_supported() &&
      can_run_neon_mul_mat_q8_0_packed_request(
          request, ::emel::kernel::detail::dtype_q8_0_x4_bl4);
}

inline bool can_use_neon_mul_mat_q8_0_packed_bl8(const event::op_mul_mat & request,
                                                 const bool neon_available) noexcept {
  return neon_available &&
      neon_q8_0_packed_bl8_supported() &&
      can_run_neon_mul_mat_q8_0_packed_request(
          request, ::emel::kernel::detail::dtype_q8_0_x4_bl8);
}

inline bool can_use_neon_mul_mat_q8_0_packed_bl8_full_groups(
    const event::op_mul_mat & request,
    const bool neon_available) noexcept {
  return can_use_neon_mul_mat_q8_0_packed_bl8(request, neon_available) &&
      (request.src0.ne[1] % ::emel::kernel::detail::quant::Q8_0_X4_ROWS) == 0u;
}

inline bool can_use_neon_mul_mat_q8_0_packed_bl8_matrix_x4(
    const event::op_mul_mat & request,
    const bool neon_available) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t rhs_rows = request.src1.ne[0];
  const uint64_t lhs_group_count = ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(m);
  const uint64_t rhs_group_count =
      ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(rhs_rows);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q8_0_x4_group_storage_bytes(k);
  const size_t dst_row_bytes = sizeof(float) * m;
  return neon_available &&
      neon_q8_0_packed_bl8_supported() &&
      k != 0u &&
      m != 0u &&
      rhs_rows == ::emel::kernel::detail::quant::Q8_0_X4_ROWS &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == rhs_rows &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      (m % ::emel::kernel::detail::quant::Q8_0_X4_ROWS) == 0u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q8_0_x4_bl8 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_0_x4_bl8 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      group_bytes != 0u &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == group_bytes &&
      request.src0.nb[2] == group_bytes * lhs_group_count &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      request.src1.nb[1] == group_bytes &&
      request.src1.nb[2] == group_bytes * rhs_group_count &&
      request.src1.nb[3] == request.src1.nb[2] &&
      request.dst.nb[0] == dst_row_bytes &&
      request.dst.nb[1] == sizeof(float) &&
      request.dst.nb[2] == dst_row_bytes * rhs_rows &&
      request.dst.nb[3] == request.dst.nb[2];
}

inline bool can_use_neon_mul_mat_q8_0_packed_bl8_tail_safe(
    const event::op_mul_mat & request,
    const bool neon_available) noexcept {
  return can_use_neon_mul_mat_q8_0_packed_bl8(request, neon_available) &&
      !can_use_neon_mul_mat_q8_0_packed_bl8_full_groups(request, neon_available);
}

inline bool neon_q6_vector_packed_supported() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  return true;
#else
  return false;
#endif
}

inline bool neon_q6_vector_prepared_q8_rhs_i8mm_supported() noexcept {
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  return true;
#else
  return false;
#endif
}

inline bool can_run_neon_mul_mat_q6_vector_packed_request(
    const event::op_mul_mat & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(k);
  return k != 0u &&
      m != 0u &&
      block_count != 0u &&
      block_count <= ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k_x8 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      request.src0.nb[0] == 1u &&
      group_bytes != 0u &&
      request.src0.nb[1] == group_bytes &&
      request.src0.nb[2] == group_bytes * group_count &&
      request.src0.nb[3] == request.src0.nb[2] &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
}

inline bool can_run_neon_mul_mat_q6_vector_packed_q8_rhs_request(
    const event::op_mul_mat & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return k != 0u &&
      m != 0u &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == m &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k_x8 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_k &&
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
      request.src1.nb[2] == rhs_row_bytes &&
      request.src1.nb[3] == rhs_row_bytes &&
      is_dense_contiguous(request.dst);
}

inline bool can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_request(
    const event::op_mul_mat & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return k != 0u &&
      m != 0u &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
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
          ::emel::kernel::detail::dtype_q8_k &&
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
      request.src1.nb[2] == rhs_row_bytes &&
      request.src1.nb[3] == rhs_row_bytes &&
      is_dense_contiguous(request.dst);
}

inline bool can_use_neon_mul_mat_q6_vector_packed(const event::op_mul_mat & request,
                                                  const bool neon_available) noexcept {
  return neon_available &&
      neon_q6_vector_packed_supported() &&
      can_run_neon_mul_mat_q6_vector_packed_request(request);
}

inline bool can_use_neon_mul_mat_q6_vector_packed_q8_rhs(const event::op_mul_mat & request,
                                                         const bool neon_available) noexcept {
  return neon_available &&
      neon_q6_vector_packed_supported() &&
      can_run_neon_mul_mat_q6_vector_packed_q8_rhs_request(request);
}

inline bool can_use_neon_mul_mat_q6_vector_prepared_q8_rhs(const event::op_mul_mat & request,
                                                           const bool neon_available) noexcept {
#if defined(__ARM_FEATURE_MATMUL_INT8)
  (void) request;
  (void) neon_available;
  return false;
#else
  return neon_available &&
      neon_q6_vector_packed_supported() &&
      can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_request(request);
#endif
}

inline bool can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm(
    const event::op_mul_mat & request,
    const bool neon_available) noexcept {
  return neon_available &&
      neon_q6_vector_prepared_q8_rhs_i8mm_supported() &&
      can_run_neon_mul_mat_q6_vector_prepared_q8_rhs_request(request);
}

inline bool can_run_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_request(
    const event::op_mul_mat_argmax & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return request.index_out != nullptr &&
      k != 0u &&
      m != 0u &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == 1u &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k_x8_q8_prepared &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_k &&
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
      request.src1.nb[2] == rhs_row_bytes &&
      request.src1.nb[3] == rhs_row_bytes &&
      ::emel::kernel::aarch64::detail::is_dense_contiguous(request.dst);
}

inline bool can_run_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_request(
    const event::op_mul_mat_argmax & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return request.index_out != nullptr &&
      k != 0u &&
      m != 0u &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == 1u &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_k &&
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
      request.src1.nb[2] == rhs_row_bytes &&
      request.src1.nb[3] == rhs_row_bytes &&
      ::emel::kernel::aarch64::detail::is_dense_contiguous(request.dst);
}

inline bool can_run_neon_mul_mat_argmax_q6_vector_packed_q8_rhs_request(
    const event::op_mul_mat_argmax & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  const size_t group_bytes =
      ::emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(k);
  const size_t rhs_row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_k, k);
  return request.index_out != nullptr &&
      k != 0u &&
      m != 0u &&
      request.src1.ne[0] == 1u &&
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == 1u &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_q6_k_x8 &&
      ::emel::kernel::detail::dtype_code(request.src1.type) ==
          ::emel::kernel::detail::dtype_q8_k &&
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
      request.src1.nb[2] == rhs_row_bytes &&
      request.src1.nb[3] == rhs_row_bytes &&
      ::emel::kernel::aarch64::detail::is_dense_contiguous(request.dst);
}

inline bool can_use_neon_mul_mat_argmax_q6_vector_packed_q8_rhs(
    const event::op_mul_mat_argmax & request,
    const bool neon_available) noexcept {
  return neon_available &&
      neon_q6_vector_packed_supported() &&
      can_run_neon_mul_mat_argmax_q6_vector_packed_q8_rhs_request(request);
}

inline bool can_use_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm(
    const event::op_mul_mat_argmax & request,
    const bool neon_available) noexcept {
  return neon_available &&
      neon_q6_vector_prepared_q8_rhs_i8mm_supported() &&
      can_run_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_request(request);
}

inline bool can_use_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm(
    const event::op_mul_mat_argmax & request,
    const bool neon_available) noexcept {
  return neon_available &&
      neon_q6_vector_prepared_q8_rhs_i8mm_supported() &&
      can_run_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_request(request);
}

template <class request_type>
inline bool can_use_neon_flash_attn_ext_f16kv_one_chunk(
    const request_type & request,
    const bool neon_available) noexcept {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  (void) request;
  (void) neon_available;
  return false;
#else
  return neon_available && ::emel::kernel::detail::can_run_flash_attn_ext(request);
#endif
}

template <class request_type>
inline bool can_run_neon_flash_attn_ext_f16kv_one_chunk_request(
    const request_type & request,
    const bool neon_available,
    const ::emel::kernel::detail::flash_attn_workspace & workspace) noexcept {
  return can_use_neon_flash_attn_ext_f16kv_one_chunk(request, neon_available) &&
      ::emel::kernel::detail::can_run_flash_attn_ext_with_workspace(request, workspace);
}

template <class tensor_type>
inline const float * tensor_row_ptr(const tensor_type & tensor,
                                    const uint64_t row1,
                                    const uint64_t row2) noexcept {
  const auto * base = static_cast<const char *>(tensor.data);
  return reinterpret_cast<const float *>(base + row1 * tensor.nb[1] + row2 * tensor.nb[2]);
}

template <class tensor_type>
inline float * tensor_row_ptr_mut(const tensor_type & tensor,
                                  const uint64_t row1,
                                  const uint64_t row2) noexcept {
  auto * base = static_cast<char *>(tensor.data);
  return reinterpret_cast<float *>(base + row1 * tensor.nb[1] + row2 * tensor.nb[2]);
}

inline void scale_f32_neon(float * data, const float scale, const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const float32x4_t scale_v = vdupq_n_f32(scale);
  uint64_t idx = 0;
  for (; idx + 4 <= count; idx += 4) {
    const float32x4_t data_v = vld1q_f32(data + idx);
    vst1q_f32(data + idx, vmulq_f32(data_v, scale_v));
  }
  for (; idx < count; ++idx) {
    data[idx] *= scale;
  }
#else
  (void) data;
  (void) scale;
  (void) count;
#endif
}

inline void axpy_f32_neon(float * dst, const float * src,
                          const float alpha, const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const float32x4_t alpha_v = vdupq_n_f32(alpha);
  uint64_t idx = 0;
  for (; idx + 4 <= count; idx += 4) {
    const float32x4_t dst_v = vld1q_f32(dst + idx);
    const float32x4_t src_v = vld1q_f32(src + idx);
    const float32x4_t prod_v = vmulq_f32(src_v, alpha_v);
    vst1q_f32(dst + idx, vaddq_f32(dst_v, prod_v));
  }
  for (; idx < count; ++idx) {
    dst[idx] += src[idx] * alpha;
  }
#else
  (void) dst;
  (void) src;
  (void) alpha;
  (void) count;
#endif
}

inline void convert_f32_to_f16_buffer_neon(const float * src,
                                           uint16_t * dst,
                                           const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    const float16x8_t fp16_0 =
        vcombine_f16(vcvt_f16_f32(vld1q_f32(src + idx + 0u)),
                     vcvt_f16_f32(vld1q_f32(src + idx + 4u)));
    const float16x8_t fp16_1 =
        vcombine_f16(vcvt_f16_f32(vld1q_f32(src + idx + 8u)),
                     vcvt_f16_f32(vld1q_f32(src + idx + 12u)));
    vst1q_u16(dst + idx + 0u, vreinterpretq_u16_f16(fp16_0));
    vst1q_u16(dst + idx + 8u, vreinterpretq_u16_f16(fp16_1));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const float16x8_t fp16 =
        vcombine_f16(vcvt_f16_f32(vld1q_f32(src + idx + 0u)),
                     vcvt_f16_f32(vld1q_f32(src + idx + 4u)));
    vst1q_u16(dst + idx + 0u, vreinterpretq_u16_f16(fp16));
  }
  for (; idx < count; ++idx) {
    dst[idx] = ::emel::kernel::detail::quant::fp32_to_fp16(src[idx]);
  }
  return;
#endif
#endif
  ::emel::kernel::detail::convert_f32_to_fp16_buffer_scalar(src, dst, count);
}

inline float dot_product_f16_f16_scores_neon(const uint16_t * lhs,
                                             const uint16_t * rhs,
                                             const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float16x8_t sum[4] = {
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
  };
  uint64_t idx = 0u;
  const auto * lhs_f16 = reinterpret_cast<const __fp16 *>(lhs);
  const auto * rhs_f16 = reinterpret_cast<const __fp16 *>(rhs);
  for (; idx + 32u <= count; idx += 32u) {
    sum[0] = vfmaq_f16(sum[0], vld1q_f16(lhs_f16 + idx + 0u), vld1q_f16(rhs_f16 + idx + 0u));
    sum[1] = vfmaq_f16(sum[1], vld1q_f16(lhs_f16 + idx + 8u), vld1q_f16(rhs_f16 + idx + 8u));
    sum[2] = vfmaq_f16(sum[2], vld1q_f16(lhs_f16 + idx + 16u), vld1q_f16(rhs_f16 + idx + 16u));
    sum[3] = vfmaq_f16(sum[3], vld1q_f16(lhs_f16 + idx + 24u), vld1q_f16(rhs_f16 + idx + 24u));
  }

  double sumf = 0.0;
  if (idx != 0u) {
    int offset = 2;
    for (int i = 0; i < offset; ++i) {
      sum[i] = vaddq_f16(sum[i], sum[offset + i]);
    }
    offset >>= 1;
    for (int i = 0; i < offset; ++i) {
      sum[i] = vaddq_f16(sum[i], sum[offset + i]);
    }

    const float32x4_t low = vcvt_f32_f16(vget_low_f16(sum[0]));
    const float32x4_t high = vcvt_f32_f16(vget_high_f16(sum[0]));
    sumf = static_cast<double>(vaddvq_f32(vaddq_f32(low, high)));
  }
  for (; idx < count; ++idx) {
    sumf += static_cast<double>(::emel::kernel::detail::quant::fp16_to_fp32(lhs[idx])) *
            static_cast<double>(::emel::kernel::detail::quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(sumf);
#endif
#endif

  float scalar_sum = 0.0f;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum += ::emel::kernel::detail::quant::fp16_to_fp32(lhs[idx]) *
                  ::emel::kernel::detail::quant::fp16_to_fp32(rhs[idx]);
  }
  return scalar_sum;
}

inline void scale_f16_buffer_neon(uint16_t * data,
                                  const float scale,
                                  const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const float rounded_scale = ::emel::kernel::detail::round_fp16_scalar(scale);
  const float16x8_t scale_v = vdupq_n_f16(static_cast<__fp16>(rounded_scale));
  uint64_t idx = 0u;
  for (; idx + 32u <= count; idx += 32u) {
    const float16x8_t data0 = vreinterpretq_f16_u16(vld1q_u16(data + idx + 0u));
    const float16x8_t data1 = vreinterpretq_f16_u16(vld1q_u16(data + idx + 8u));
    const float16x8_t data2 = vreinterpretq_f16_u16(vld1q_u16(data + idx + 16u));
    const float16x8_t data3 = vreinterpretq_f16_u16(vld1q_u16(data + idx + 24u));
    vst1q_u16(data + idx + 0u, vreinterpretq_u16_f16(vmulq_f16(data0, scale_v)));
    vst1q_u16(data + idx + 8u, vreinterpretq_u16_f16(vmulq_f16(data1, scale_v)));
    vst1q_u16(data + idx + 16u, vreinterpretq_u16_f16(vmulq_f16(data2, scale_v)));
    vst1q_u16(data + idx + 24u, vreinterpretq_u16_f16(vmulq_f16(data3, scale_v)));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const float16x8_t data_v = vreinterpretq_f16_u16(vld1q_u16(data + idx));
    const float16x8_t scaled = vmulq_f16(data_v, scale_v);
    vst1q_u16(data + idx, vreinterpretq_u16_f16(scaled));
  }
  for (; idx < count; ++idx) {
    const float rounded_value = ::emel::kernel::detail::quant::fp16_to_fp32(data[idx]);
    data[idx] = ::emel::kernel::detail::quant::fp32_to_fp16(
        ::emel::kernel::detail::round_fp16_scalar(rounded_value * rounded_scale));
  }
  return;
#endif
#endif
  ::emel::kernel::detail::scale_f16_buffer_scalar(data, scale, count);
}

inline void axpy_f16_buffer_neon(uint16_t * dst,
                                 const uint16_t * src,
                                 const float alpha,
                                 const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const float rounded_alpha = ::emel::kernel::detail::round_fp16_scalar(alpha);
  const float16x8_t alpha_v = vdupq_n_f16(static_cast<__fp16>(rounded_alpha));
  auto * dst_f16 = reinterpret_cast<__fp16 *>(dst);
  const auto * src_f16 = reinterpret_cast<const __fp16 *>(src);
  uint64_t idx = 0u;
  for (; idx + 32u <= count; idx += 32u) {
    const float16x8_t dst0 = vld1q_f16(dst_f16 + idx + 0u);
    const float16x8_t src0 = vld1q_f16(src_f16 + idx + 0u);
    const float16x8_t dst1 = vld1q_f16(dst_f16 + idx + 8u);
    const float16x8_t src1 = vld1q_f16(src_f16 + idx + 8u);
    const float16x8_t dst2 = vld1q_f16(dst_f16 + idx + 16u);
    const float16x8_t src2 = vld1q_f16(src_f16 + idx + 16u);
    const float16x8_t dst3 = vld1q_f16(dst_f16 + idx + 24u);
    const float16x8_t src3 = vld1q_f16(src_f16 + idx + 24u);
    vst1q_f16(dst_f16 + idx + 0u, vfmaq_f16(dst0, src0, alpha_v));
    vst1q_f16(dst_f16 + idx + 8u, vfmaq_f16(dst1, src1, alpha_v));
    vst1q_f16(dst_f16 + idx + 16u, vfmaq_f16(dst2, src2, alpha_v));
    vst1q_f16(dst_f16 + idx + 24u, vfmaq_f16(dst3, src3, alpha_v));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const float16x8_t dst_v = vld1q_f16(dst_f16 + idx);
    const float16x8_t src_v = vld1q_f16(src_f16 + idx);
    const float16x8_t out = vfmaq_f16(dst_v, src_v, alpha_v);
    vst1q_f16(dst_f16 + idx, out);
  }
  for (; idx < count; ++idx) {
    const float rounded_dst = ::emel::kernel::detail::quant::fp16_to_fp32(dst[idx]);
    const float rounded_src = ::emel::kernel::detail::quant::fp16_to_fp32(src[idx]);
    dst[idx] = ::emel::kernel::detail::quant::fp32_to_fp16(
        ::emel::kernel::detail::round_fp16_scalar(rounded_dst + rounded_src * rounded_alpha));
  }
  return;
#endif
#endif
  ::emel::kernel::detail::axpy_f16_buffer_scalar(dst, src, alpha, count);
}

inline void convert_f16_buffer_to_f32_neon(const uint16_t * src,
                                           float * dst,
                                           const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    const float16x8_t fp16_0 = vreinterpretq_f16_u16(vld1q_u16(src + idx + 0u));
    const float16x8_t fp16_1 = vreinterpretq_f16_u16(vld1q_u16(src + idx + 8u));
    vst1q_f32(dst + idx + 0u, vcvt_f32_f16(vget_low_f16(fp16_0)));
    vst1q_f32(dst + idx + 4u, vcvt_f32_f16(vget_high_f16(fp16_0)));
    vst1q_f32(dst + idx + 8u, vcvt_f32_f16(vget_low_f16(fp16_1)));
    vst1q_f32(dst + idx + 12u, vcvt_f32_f16(vget_high_f16(fp16_1)));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const float16x8_t fp16 = vreinterpretq_f16_u16(vld1q_u16(src + idx));
    vst1q_f32(dst + idx + 0u, vcvt_f32_f16(vget_low_f16(fp16)));
    vst1q_f32(dst + idx + 4u, vcvt_f32_f16(vget_high_f16(fp16)));
  }
  for (; idx < count; ++idx) {
    dst[idx] = ::emel::kernel::detail::quant::fp16_to_fp32(src[idx]);
  }
  return;
#endif
#endif
  ::emel::kernel::detail::convert_f16_buffer_to_f32_scalar(src, dst, count);
}

template <class request_type>
inline void prepare_flash_attn_ext_f16kv_one_chunk_workspace_neon(
    const request_type & request,
    ::emel::kernel::detail::flash_attn_workspace & workspace) noexcept {
  const uint64_t kv_tokens = ::emel::kernel::detail::flash_attn_active_tokens(request);
  const bool reusing = workspace.prepared_tokens == kv_tokens;
  workspace.reuse_count += static_cast<uint64_t>(reusing);
  workspace.prepared_tokens = kv_tokens;
}

template <class request_type>
inline void run_flash_attn_ext_f16kv_one_chunk_neon_unchecked(
    const request_type & request,
    ::emel::kernel::detail::flash_attn_workspace & workspace) noexcept {
  const uint64_t kv_tokens = ::emel::kernel::detail::flash_attn_active_tokens(request);
  prepare_flash_attn_ext_f16kv_one_chunk_workspace_neon(request, workspace);
  const uint64_t head_dim = request.src0.ne[0];
  const uint64_t head_count = request.src0.ne[2];
  const uint64_t kv_head_count = request.src1.ne[2];
  const float scale = ::emel::kernel::detail::flash_attn_scale(request);
  const uint64_t n_rep = head_count / kv_head_count;
  for (uint64_t head = 0; head < head_count; ++head) {
    const uint64_t kv_head = head / n_rep;
    const float * q = ::emel::kernel::detail::tensor_row_ptr(request.src0, 0u, head);
    uint16_t * accum = workspace.accum_buffer_f16.data();
    float * dst = ::emel::kernel::detail::tensor_row_ptr_mut(request.dst, 0u, head);

    convert_f32_to_f16_buffer_neon(q, workspace.q_buffer_f16.data(), head_dim);
    std::memset(accum, 0, sizeof(uint16_t) * head_dim);

    const auto * k_head_base =
        static_cast<const char *>(request.src1.data) + kv_head * request.src1.nb[2];
    const auto * v_head_base =
        static_cast<const char *>(request.src2.data) + kv_head * request.src2.nb[2];
    const uint64_t k_stride = request.src1.nb[1];
    const uint64_t v_stride = request.src2.nb[1];
    const char * k_ptr_bytes = k_head_base;
    const char * v_ptr_bytes = v_head_base;

    float score_sum = 0.0f;
    float max_score = -std::numeric_limits<float>::infinity();
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const uint16_t * k = reinterpret_cast<const uint16_t *>(k_ptr_bytes);
      const float score =
          dot_product_f16_f16_scores_neon(workspace.q_buffer_f16.data(), k, head_dim) * scale;
      const float old_max = max_score;
      float max_scale = 1.0f;
      float value_scale = 1.0f;
      if (score > max_score) {
        max_score = score;
        max_scale = std::exp(old_max - max_score);
        scale_f16_buffer_neon(accum, max_scale, head_dim);
      } else {
        value_scale = std::exp(score - max_score);
      }

      const uint16_t * v = reinterpret_cast<const uint16_t *>(v_ptr_bytes);
      axpy_f16_buffer_neon(accum, v, value_scale, head_dim);
      score_sum = score_sum * max_scale + value_scale;

      k_ptr_bytes += k_stride;
      v_ptr_bytes += v_stride;
    }

    convert_f16_buffer_to_f32_neon(accum, dst, head_dim);
    if (score_sum == 0.0f) {
      std::fill_n(dst, head_dim, 0.0f);
    } else {
      scale_f32_neon(dst, 1.0f / score_sum, head_dim);
    }
  }
}

template <class request_type>
inline void run_flash_attn_ext_neon_unchecked(
    const request_type & request,
    ::emel::kernel::detail::flash_attn_workspace & workspace) noexcept {
  run_flash_attn_ext_f16kv_one_chunk_neon_unchecked(request, workspace);
}

template <class request_type>
inline bool run_flash_attn_ext_neon(const request_type & request,
                                    const bool neon_available,
                                    ::emel::kernel::detail::flash_attn_workspace & workspace) noexcept {
  if (!can_run_neon_flash_attn_ext_f16kv_one_chunk_request(
          request, neon_available, workspace)) {
    return false;
  }
  run_flash_attn_ext_neon_unchecked(request, workspace);
  return true;
}

inline bool execute_neon_dup(const event::op_dup & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    vst1q_f32(dst + i, v);
  }
  for (; i < count; ++i) {
    dst[i] = src[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_add(const event::op_add & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * lhs = static_cast<const float *>(request.src0.data);
  const float * rhs = static_cast<const float *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t a = vld1q_f32(lhs + i);
    const float32x4_t b = vld1q_f32(rhs + i);
    vst1q_f32(dst + i, vaddq_f32(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] + rhs[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_sub(const event::op_sub & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * lhs = static_cast<const float *>(request.src0.data);
  const float * rhs = static_cast<const float *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t a = vld1q_f32(lhs + i);
    const float32x4_t b = vld1q_f32(rhs + i);
    vst1q_f32(dst + i, vsubq_f32(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] - rhs[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_mul(const event::op_mul & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * lhs = static_cast<const float *>(request.src0.data);
  const float * rhs = static_cast<const float *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t a = vld1q_f32(lhs + i);
    const float32x4_t b = vld1q_f32(rhs + i);
    vst1q_f32(dst + i, vmulq_f32(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] * rhs[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_div(const event::op_div & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * lhs = static_cast<const float *>(request.src0.data);
  const float * rhs = static_cast<const float *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
#if defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    const float32x4_t a = vld1q_f32(lhs + i);
    const float32x4_t b = vld1q_f32(rhs + i);
    vst1q_f32(dst + i, vdivq_f32(a, b));
  }
#else
  for (; i + 4 <= count; i += 4) {
    const float32x4_t a = vld1q_f32(lhs + i);
    const float32x4_t b = vld1q_f32(rhs + i);
    float32x4_t recip = vrecpeq_f32(b);
    recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(b, recip), recip);
    vst1q_f32(dst + i, vmulq_f32(a, recip));
  }
#endif
  for (; i < count; ++i) {
    dst[i] = lhs[i] / rhs[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_sqr(const event::op_sqr & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    vst1q_f32(dst + i, vmulq_f32(v, v));
  }
  for (; i < count; ++i) {
    dst[i] = src[i] * src[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_sqrt(const event::op_sqrt & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
#if defined(__aarch64__)
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    vst1q_f32(dst + i, vsqrtq_f32(v));
  }
#endif
  for (; i < count; ++i) {
    dst[i] = std::sqrt(src[i]);
  }
  return true;
#else
  (void) request;
  return false;
#endif
}

inline uint8x16x2_t load_u8x16x2(const uint8_t * ptr) noexcept {
  uint8x16x2_t out{};
  out.val[0] = vld1q_u8(ptr);
  out.val[1] = vld1q_u8(ptr + 16);
  return out;
}

inline int8x16x4_t load_s8x16x4(const int8_t * ptr) noexcept {
  int8x16x4_t out{};
  out.val[0] = vld1q_s8(ptr + 0);
  out.val[1] = vld1q_s8(ptr + 16);
  out.val[2] = vld1q_s8(ptr + 32);
  out.val[3] = vld1q_s8(ptr + 48);
  return out;
}

inline float dot_q2_k_q8_k_block_neon(const ::emel::kernel::detail::quant::block_q2_k & lhs,
                                      const ::emel::kernel::detail::quant::block_q8_k & rhs)
    noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  return ::emel::kernel::detail::dot_q2_k_q8_k_block_scalar(lhs, rhs);
#else
  const uint8x16_t m3 = vdupq_n_u8(0x03u);
  const uint8x16_t m4 = vdupq_n_u8(0x0fu);
  const int32x4_t zero = vdupq_n_s32(0);

  int8x16x2_t q2bytes{};
  uint8_t scales_buf[16] = {};
  float sum = 0.0f;

  const float d = rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d);
  const float dmin = -rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.dmin);
  const uint8_t * q2 = lhs.qs.data();
  const int8_t * q8 = rhs.qs.data();
  const uint8_t * scales_ptr = lhs.scales.data();

  const uint8x16_t mins_and_scales = vld1q_u8(scales_ptr);
  const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
  vst1q_u8(scales_buf, scales);

  const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
  const int16x8_t q8sums0 = vld1q_s16(rhs.bsums.data());
  const int16x8_t q8sums1 = vld1q_s16(rhs.bsums.data() + 8);
  const int16x8_t mins16_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
  const int16x8_t mins16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
  const int32x4_t s0 = vaddq_s32(
      vmull_s16(vget_low_s16(mins16_lo), vget_low_s16(q8sums0)),
      vmull_s16(vget_high_s16(mins16_lo), vget_high_s16(q8sums0)));
  const int32x4_t s1 = vaddq_s32(
      vmull_s16(vget_low_s16(mins16_hi), vget_low_s16(q8sums1)),
      vmull_s16(vget_high_s16(mins16_hi), vget_high_s16(q8sums1)));
  sum += dmin * static_cast<float>(vaddvq_s32(vaddq_s32(s0, s1)));

  int isum = 0;
  int scale_index = 0;
  for (uint64_t j = 0; j < (::emel::kernel::detail::quant::QK_K / 128); ++j) {
    const uint8x16x2_t q2bits = load_u8x16x2(q2);
    q2 += 32;

    {
      const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
          scales_buf[scale_index + 0];
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
          scales_buf[scale_index + 1];
    }
    {
      const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], 2), m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], 2), m3));
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
          scales_buf[scale_index + 2];
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
          scales_buf[scale_index + 3];
    }
    {
      const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], 4), m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], 4), m3));
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
          scales_buf[scale_index + 4];
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
          scales_buf[scale_index + 5];
    }
    {
      const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
      q8 += 32;
      q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], 6), m3));
      q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], 6), m3));
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
          scales_buf[scale_index + 6];
      isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
          scales_buf[scale_index + 7];
    }
    scale_index += 8;
  }

  return sum + d * static_cast<float>(isum);
#endif
}

inline float dot_q2_k_q8_k_row_neon(const ::emel::kernel::detail::quant::block_q2_k * lhs,
                                    const ::emel::kernel::detail::quant::block_q8_k * rhs,
                                    const uint64_t block_count) noexcept {
  return ::emel::kernel::detail::dot_q2_k_q8_k_row_scalar(lhs, rhs, block_count);
}

inline float dot_q3_k_q8_k_block_neon(const ::emel::kernel::detail::quant::block_q3_k & lhs,
                                      const ::emel::kernel::detail::quant::block_q8_k & rhs)
    noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  return ::emel::kernel::detail::dot_q3_k_q8_k_block_scalar(lhs, rhs);
#else
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  uint32_t scale_words[4] = {};
  std::memcpy(scale_words, lhs.scales.data(), lhs.scales.size());
  const uint32_t tmp = scale_words[2];
  scale_words[3] = ((scale_words[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
  scale_words[2] = ((scale_words[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
  scale_words[1] = (scale_words[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
  scale_words[0] = (scale_words[0] & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
  auto * scales = reinterpret_cast<int8_t *>(scale_words);
  for (uint64_t j = 0; j < 16; ++j) {
    scales[j] = static_cast<int8_t>(scales[j] - 32);
  }

  const uint8x16_t m3b = vdupq_n_u8(0x03u);
  const int32x4_t zero = vdupq_n_s32(0);
  const uint8x16_t m0 = vdupq_n_u8(1u);
  const uint8x16_t m1 = vshlq_n_u8(m0, 1);
  const uint8x16_t m2 = vshlq_n_u8(m0, 2);
  const uint8x16_t m3 = vshlq_n_u8(m0, 3);

  const uint8_t * q3 = lhs.qs.data();
  const uint8_t * qh = lhs.hmask.data();
  const int8_t * q8 = rhs.qs.data();

  uint8x16x2_t qhbits = load_u8x16x2(qh);
  uint8x16x4_t q3h{};
  int8x16x4_t q3bytes{};
  int32_t isum = 0;
  int scale_index = 0;
  for (uint64_t j = 0; j < (::emel::kernel::detail::quant::QK_K / 128); ++j) {
    const uint8x16x2_t q3bits = load_u8x16x2(q3);
    q3 += 32;
    const int8x16x4_t q8bytes_1 = load_s8x16x4(q8);
    q8 += 64;
    const int8x16x4_t q8bytes_2 = load_s8x16x4(q8);
    q8 += 64;

    q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
    q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
    q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
    q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

    q3bytes.val[0] =
        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
    q3bytes.val[1] =
        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
    q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)),
                              vreinterpretq_s8_u8(q3h.val[2]));
    q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)),
                              vreinterpretq_s8_u8(q3h.val[3]));

    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[0], q8bytes_1.val[0])) * scales[scale_index + 0];
    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[1], q8bytes_1.val[1])) * scales[scale_index + 1];
    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[2], q8bytes_1.val[2])) * scales[scale_index + 2];
    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[3], q8bytes_1.val[3])) * scales[scale_index + 3];
    scale_index += 4;

    q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
    q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
    q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
    q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

    q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)),
                              vreinterpretq_s8_u8(q3h.val[0]));
    q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)),
                              vreinterpretq_s8_u8(q3h.val[1]));
    q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)),
                              vreinterpretq_s8_u8(q3h.val[2]));
    q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)),
                              vreinterpretq_s8_u8(q3h.val[3]));

    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[0], q8bytes_2.val[0])) * scales[scale_index + 0];
    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[1], q8bytes_2.val[1])) * scales[scale_index + 1];
    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[2], q8bytes_2.val[2])) * scales[scale_index + 2];
    isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[3], q8bytes_2.val[3])) * scales[scale_index + 3];
    scale_index += 4;

    if (j == 0) {
      qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
      qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
    }
  }

  const float d = ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d) * rhs.d;
  return d * static_cast<float>(isum);
#endif
}

inline float dot_q3_k_q8_k_row_neon(const ::emel::kernel::detail::quant::block_q3_k * lhs,
                                    const ::emel::kernel::detail::quant::block_q8_k * rhs,
                                    const uint64_t block_count) noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  return ::emel::kernel::detail::dot_q3_k_q8_k_row_scalar(lhs, rhs, block_count);
#else
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  uint32_t aux[3] = {};
  uint32_t utmp[4] = {};

  const uint8x16_t m3b = vdupq_n_u8(0x03u);
  const int32x4_t zero = vdupq_n_s32(0);
  const uint8x16_t m0 = vdupq_n_u8(1u);
  const uint8x16_t m1 = vshlq_n_u8(m0, 1);
  const uint8x16_t m2 = vshlq_n_u8(m0, 2);
  const uint8x16_t m3 = vshlq_n_u8(m0, 3);
  const int8_t m32 = 32;

  int8x16x4_t q3bytes{};
  float sum = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    const float d = rhs[block].d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d);

    const uint8_t * q3 = lhs[block].qs.data();
    const uint8_t * qh = lhs[block].hmask.data();
    const int8_t * q8 = rhs[block].qs.data();

    uint8x16x2_t qhbits = load_u8x16x2(qh);
    uint8x16x4_t q3h{};
    int32_t isum = 0;

    std::memcpy(aux, lhs[block].scales.data(), lhs[block].scales.size());
    utmp[3] = ((aux[1] >> 4u) & kmask2) | (((aux[2] >> 6u) & kmask1) << 4u);
    utmp[2] = ((aux[0] >> 4u) & kmask2) | (((aux[2] >> 4u) & kmask1) << 4u);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2u) & kmask1) << 4u);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0u) & kmask1) << 4u);

    int8_t * scale = reinterpret_cast<int8_t *>(utmp);
    for (uint64_t j = 0; j < 16; ++j) {
      scale[j] = static_cast<int8_t>(scale[j] - m32);
    }

    for (uint64_t j = 0; j < (::emel::kernel::detail::quant::QK_K / 128); ++j) {
      const uint8x16x2_t q3bits = load_u8x16x2(q3);
      q3 += 32;
      const int8x16x4_t q8bytes_1 = load_s8x16x4(q8);
      q8 += 64;
      const int8x16x4_t q8bytes_2 = load_s8x16x4(q8);
      q8 += 64;

      q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
      q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
      q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
      q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

      q3bytes.val[0] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] =
          vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)),
                                vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)),
                                vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];
      scale += 4;

      q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
      q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
      q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
      q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

      q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)),
                                vreinterpretq_s8_u8(q3h.val[0]));
      q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)),
                                vreinterpretq_s8_u8(q3h.val[1]));
      q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)),
                                vreinterpretq_s8_u8(q3h.val[2]));
      q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)),
                                vreinterpretq_s8_u8(q3h.val[3]));

      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
      isum += vaddvq_s32(vdotq_s32(zero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];
      scale += 4;

      if (j == 0) {
        qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
        qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
      }
    }

    sum += d * static_cast<float>(isum);
  }

  return sum;
#endif
}

inline float dot_q6_k_q8_k_block_neon(const ::emel::kernel::detail::quant::block_q6_k & lhs,
                                      const ::emel::kernel::detail::quant::block_q8_k & rhs)
    noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  return ::emel::kernel::detail::dot_q6_k_q8_k_block_scalar(lhs, rhs);
#else
  const uint8x16_t m4b = vdupq_n_u8(0x0fu);
  const int32x4_t zero = vdupq_n_s32(0);
  const uint8x16_t mone = vdupq_n_u8(3u);

  int8x16x4_t q6bytes{};
  uint8x16x4_t q6h{};
  const uint8_t * q6 = lhs.ql.data();
  const uint8_t * qh = lhs.qh.data();
  const int8_t * q8 = rhs.qs.data();
  const int8_t * scale = lhs.scales.data();

  const int16x8_t q8sums0 = vld1q_s16(rhs.bsums.data());
  const int16x8_t q8sums1 = vld1q_s16(rhs.bsums.data() + 8);
  const int8x16_t scales_s8 = vld1q_s8(scale);
  const int16x8_t q6scales0 = vmovl_s8(vget_low_s8(scales_s8));
  const int16x8_t q6scales1 = vmovl_s8(vget_high_s8(scales_s8));
  const int32x4_t prod = vaddq_s32(
      vaddq_s32(vmull_s16(vget_low_s16(q8sums0), vget_low_s16(q6scales0)),
                vmull_s16(vget_high_s16(q8sums0), vget_high_s16(q6scales0))),
      vaddq_s32(vmull_s16(vget_low_s16(q8sums1), vget_low_s16(q6scales1)),
                vmull_s16(vget_high_s16(q8sums1), vget_high_s16(q6scales1))));
  const int32_t sum_mins = vaddvq_s32(prod);

  int32_t isum = 0;
  for (uint64_t j = 0; j < (::emel::kernel::detail::quant::QK_K / 128); ++j) {
    const uint8x16x2_t qhbits = load_u8x16x2(qh);
    qh += 32;
    uint8x16x4_t q6bits{};
    q6bits.val[0] = vld1q_u8(q6 + 0);
    q6bits.val[1] = vld1q_u8(q6 + 16);
    q6bits.val[2] = vld1q_u8(q6 + 32);
    q6bits.val[3] = vld1q_u8(q6 + 48);
    q6 += 64;
    const int8x16x4_t q8bytes_1 = load_s8x16x4(q8);
    q8 += 64;

    q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
    q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
    uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
    q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits.val[1], 2);
    q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

    q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
    q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
    q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
    q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

    isum += vaddvq_s32(vdotq_s32(zero, q6bytes.val[0], q8bytes_1.val[0])) * scale[0] +
        vaddvq_s32(vdotq_s32(zero, q6bytes.val[1], q8bytes_1.val[1])) * scale[1] +
        vaddvq_s32(vdotq_s32(zero, q6bytes.val[2], q8bytes_1.val[2])) * scale[2] +
        vaddvq_s32(vdotq_s32(zero, q6bytes.val[3], q8bytes_1.val[3])) * scale[3];
    scale += 4;

    const int8x16x4_t q8bytes_2 = load_s8x16x4(q8);
    q8 += 64;
    shifted = vshrq_n_u8(qhbits.val[0], 4);
    q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits.val[1], 4);
    q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits.val[0], 6);
    q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
    shifted = vshrq_n_u8(qhbits.val[1], 6);
    q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

    q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
    q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
    q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
    q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

    isum += vaddvq_s32(vdotq_s32(zero, q6bytes.val[0], q8bytes_2.val[0])) * scale[0] +
        vaddvq_s32(vdotq_s32(zero, q6bytes.val[1], q8bytes_2.val[1])) * scale[1] +
        vaddvq_s32(vdotq_s32(zero, q6bytes.val[2], q8bytes_2.val[2])) * scale[2] +
        vaddvq_s32(vdotq_s32(zero, q6bytes.val[3], q8bytes_2.val[3])) * scale[3];
    scale += 4;
  }

  const float d = ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d) * rhs.d;
  return d * static_cast<float>(isum - 32 * sum_mins);
#endif
}

inline float dot_q6_k_q8_k_row_neon(const ::emel::kernel::detail::quant::block_q6_k * lhs,
                                    const ::emel::kernel::detail::quant::block_q8_k * rhs,
                                    const uint64_t block_count) noexcept {
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q6_k_q8_k_block_neon(lhs[block], rhs[block]);
  }
  return sum;
}

inline int32_t horizontal_sum_s32_neon(const int32x4_t value) noexcept {
#if defined(__aarch64__)
  return vaddvq_s32(value);
#else
  const int32x2_t pair = vadd_s32(vget_low_s32(value), vget_high_s32(value));
  return vget_lane_s32(vpadd_s32(pair, pair), 0);
#endif
}

inline int32_t dot_q8_0_q8_0_block_sum_neon(
    const ::emel::kernel::detail::quant::block_q8_0 & lhs,
    const ::emel::kernel::detail::quant::block_q8_0 & rhs) noexcept {
#if defined(__ARM_FEATURE_DOTPROD)
  const int32x4_t zero = vdupq_n_s32(0);
  const int8x16_t lhs0 = vld1q_s8(lhs.qs.data());
  const int8x16_t rhs0 = vld1q_s8(rhs.qs.data());
  const int8x16_t lhs1 = vld1q_s8(lhs.qs.data() + 16);
  const int8x16_t rhs1 = vld1q_s8(rhs.qs.data() + 16);
  const int32x4_t acc = vaddq_s32(
      vdotq_s32(zero, lhs0, rhs0),
      vdotq_s32(zero, lhs1, rhs1));
#else
  const int8x16_t lhs0 = vld1q_s8(lhs.qs.data());
  const int8x16_t rhs0 = vld1q_s8(rhs.qs.data());
  const int8x16_t lhs1 = vld1q_s8(lhs.qs.data() + 16);
  const int8x16_t rhs1 = vld1q_s8(rhs.qs.data() + 16);
  const int16x8_t prod0_lo = vmull_s8(vget_low_s8(lhs0), vget_low_s8(rhs0));
  const int16x8_t prod0_hi = vmull_s8(vget_high_s8(lhs0), vget_high_s8(rhs0));
  const int16x8_t prod1_lo = vmull_s8(vget_low_s8(lhs1), vget_low_s8(rhs1));
  const int16x8_t prod1_hi = vmull_s8(vget_high_s8(lhs1), vget_high_s8(rhs1));
  const int32x4_t acc = vaddq_s32(
      vaddq_s32(vpaddlq_s16(prod0_lo), vpaddlq_s16(prod0_hi)),
      vaddq_s32(vpaddlq_s16(prod1_lo), vpaddlq_s16(prod1_hi)));
#endif
  return horizontal_sum_s32_neon(acc);
}

inline float dot_q8_0_q8_0_row_neon(const ::emel::kernel::detail::quant::block_q8_0 * lhs,
                                    const ::emel::kernel::detail::quant::block_q8_0 * rhs,
                                    const uint64_t block_count) noexcept {
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const int32_t sumi = dot_q8_0_q8_0_block_sum_neon(lhs[block], rhs[block]);
    sum += static_cast<float>(sumi) *
        (::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d) *
         ::emel::kernel::detail::quant::fp16_to_fp32(rhs[block].d));
  }
  return sum;
}

inline void store_q8_0_x4_results(float * dst,
                                  const uint64_t row_base,
                                  const uint64_t total_rows,
                                  const float32x4_t values) noexcept {
  if ((row_base + 4u) <= total_rows) {
    vst1q_f32(dst + row_base, values);
    return;
  }

  alignas(16) float lanes[4] = {};
  vst1q_f32(lanes, values);
  const uint64_t remaining = total_rows - row_base;
  const uint64_t store_count = std::min<uint64_t>(remaining, 4u);
  for (uint64_t lane = 0; lane < store_count; ++lane) {
    dst[row_base + lane] = lanes[lane];
  }
}

inline int32_t q6_k_sum_mins_neon(const ::emel::kernel::detail::quant::block_q6_k & lhs,
                                  const int16x8_t q8sums0,
                                  const int16x8_t q8sums1) noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  (void) lhs;
  (void) q8sums0;
  (void) q8sums1;
  return 0;
#else
  const int8x16_t scales_s8 = vld1q_s8(lhs.scales.data());
  const int16x8_t q6scales0 = vmovl_s8(vget_low_s8(scales_s8));
  const int16x8_t q6scales1 = vmovl_s8(vget_high_s8(scales_s8));
  const int32x4_t prod = vaddq_s32(
      vaddq_s32(vmull_s16(vget_low_s16(q8sums0), vget_low_s16(q6scales0)),
                vmull_s16(vget_high_s16(q8sums0), vget_high_s16(q6scales0))),
      vaddq_s32(vmull_s16(vget_low_s16(q8sums1), vget_low_s16(q6scales1)),
                vmull_s16(vget_high_s16(q8sums1), vget_high_s16(q6scales1))));
  return vaddvq_s32(prod);
#endif
}

inline int32_t q6_k_lower_half_dot_neon(const uint8x16x2_t & qhbits,
                                        const uint8x16x4_t & q6bits,
                                        const int8x16x4_t & q8bytes,
                                        const int8_t * scale) noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  (void) qhbits;
  (void) q6bits;
  (void) q8bytes;
  (void) scale;
  return 0;
#else
  const uint8x16_t m4b = vdupq_n_u8(0x0fu);
  const uint8x16_t mone = vdupq_n_u8(3u);
  const int32x4_t zero = vdupq_n_s32(0);
  const uint8x16_t q6h0 = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
  const uint8x16_t q6h1 = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
  const uint8x16_t q6h2 =
      vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.val[0], 2)), 4);
  const uint8x16_t q6h3 =
      vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.val[1], 2)), 4);
  const int8x16_t q6bytes0 =
      vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h0));
  const int8x16_t q6bytes1 =
      vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h1));
  const int8x16_t q6bytes2 =
      vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h2));
  const int8x16_t q6bytes3 =
      vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h3));
  return vaddvq_s32(vdotq_s32(zero, q6bytes0, q8bytes.val[0])) * scale[0] +
      vaddvq_s32(vdotq_s32(zero, q6bytes1, q8bytes.val[1])) * scale[1] +
      vaddvq_s32(vdotq_s32(zero, q6bytes2, q8bytes.val[2])) * scale[2] +
      vaddvq_s32(vdotq_s32(zero, q6bytes3, q8bytes.val[3])) * scale[3];
#endif
}

inline int32_t q6_k_upper_half_dot_neon(const uint8x16x2_t & qhbits,
                                        const uint8x16x4_t & q6bits,
                                        const int8x16x4_t & q8bytes,
                                        const int8_t * scale) noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  (void) qhbits;
  (void) q6bits;
  (void) q8bytes;
  (void) scale;
  return 0;
#else
  const uint8x16_t mone = vdupq_n_u8(3u);
  const int32x4_t zero = vdupq_n_s32(0);
  const uint8x16_t q6h0 =
      vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.val[0], 4)), 4);
  const uint8x16_t q6h1 =
      vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.val[1], 4)), 4);
  const uint8x16_t q6h2 =
      vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.val[0], 6)), 4);
  const uint8x16_t q6h3 =
      vshlq_n_u8(vandq_u8(mone, vshrq_n_u8(qhbits.val[1], 6)), 4);
  const int8x16_t q6bytes0 =
      vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h0));
  const int8x16_t q6bytes1 =
      vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h1));
  const int8x16_t q6bytes2 =
      vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h2));
  const int8x16_t q6bytes3 =
      vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h3));
  return vaddvq_s32(vdotq_s32(zero, q6bytes0, q8bytes.val[0])) * scale[0] +
      vaddvq_s32(vdotq_s32(zero, q6bytes1, q8bytes.val[1])) * scale[1] +
      vaddvq_s32(vdotq_s32(zero, q6bytes2, q8bytes.val[2])) * scale[2] +
      vaddvq_s32(vdotq_s32(zero, q6bytes3, q8bytes.val[3])) * scale[3];
#endif
}

inline void dot_q6_k_q8_k_4rows_neon(const ::emel::kernel::detail::quant::block_q6_k * lhs0,
                                     const ::emel::kernel::detail::quant::block_q6_k * lhs1,
                                     const ::emel::kernel::detail::quant::block_q6_k * lhs2,
                                     const ::emel::kernel::detail::quant::block_q6_k * lhs3,
                                     const ::emel::kernel::detail::quant::block_q8_k * rhs,
                                     const uint64_t block_count,
                                     float * out) noexcept {
#if !defined(__ARM_FEATURE_DOTPROD)
  out[0] = dot_q6_k_q8_k_row_neon(lhs0, rhs, block_count);
  out[1] = dot_q6_k_q8_k_row_neon(lhs1, rhs, block_count);
  out[2] = dot_q6_k_q8_k_row_neon(lhs2, rhs, block_count);
  out[3] = dot_q6_k_q8_k_row_neon(lhs3, rhs, block_count);
#else
  std::array<float, 4u> sums = {};
  const std::array<const ::emel::kernel::detail::quant::block_q6_k *, 4u> lhs_rows = {
      lhs0, lhs1, lhs2, lhs3};

  for (uint64_t block = 0; block < block_count; ++block) {
    const int16x8_t q8sums0 = vld1q_s16(rhs[block].bsums.data());
    const int16x8_t q8sums1 = vld1q_s16(rhs[block].bsums.data() + 8);
    std::array<int32_t, 4u> sum_mins = {};
    for (size_t row = 0; row < lhs_rows.size(); ++row) {
      sum_mins[row] = q6_k_sum_mins_neon(lhs_rows[row][block], q8sums0, q8sums1);
    }

    std::array<int32_t, 4u> isum = {};
    std::array<const uint8_t *, 4u> ql_ptrs = {
        lhs0[block].ql.data(), lhs1[block].ql.data(), lhs2[block].ql.data(), lhs3[block].ql.data()};
    std::array<const uint8_t *, 4u> qh_ptrs = {
        lhs0[block].qh.data(), lhs1[block].qh.data(), lhs2[block].qh.data(), lhs3[block].qh.data()};
    std::array<const int8_t *, 4u> scale_ptrs = {
        lhs0[block].scales.data(),
        lhs1[block].scales.data(),
        lhs2[block].scales.data(),
        lhs3[block].scales.data(),
    };
    const int8_t * q8 = rhs[block].qs.data();

    for (uint64_t j = 0; j < (::emel::kernel::detail::quant::QK_K / 128u); ++j) {
      const int8x16x4_t q8bytes_1 = load_s8x16x4(q8);
      q8 += 64;
      std::array<uint8x16x2_t, 4u> qhbits = {};
      std::array<uint8x16x4_t, 4u> q6bits = {};
      for (size_t row = 0; row < lhs_rows.size(); ++row) {
        qhbits[row] = load_u8x16x2(qh_ptrs[row]);
        q6bits[row].val[0] = vld1q_u8(ql_ptrs[row] + 0);
        q6bits[row].val[1] = vld1q_u8(ql_ptrs[row] + 16);
        q6bits[row].val[2] = vld1q_u8(ql_ptrs[row] + 32);
        q6bits[row].val[3] = vld1q_u8(ql_ptrs[row] + 48);
        isum[row] += q6_k_lower_half_dot_neon(
            qhbits[row], q6bits[row], q8bytes_1, scale_ptrs[row]);
        scale_ptrs[row] += 4;
      }

      const int8x16x4_t q8bytes_2 = load_s8x16x4(q8);
      q8 += 64;
      for (size_t row = 0; row < lhs_rows.size(); ++row) {
        isum[row] += q6_k_upper_half_dot_neon(
            qhbits[row], q6bits[row], q8bytes_2, scale_ptrs[row]);
        scale_ptrs[row] += 4;
        ql_ptrs[row] += 64;
        qh_ptrs[row] += 32;
      }
    }

    const float rhs_d = rhs[block].d;
    for (size_t row = 0; row < lhs_rows.size(); ++row) {
      const float d =
          ::emel::kernel::detail::quant::fp16_to_fp32(lhs_rows[row][block].d) * rhs_d;
      sums[row] += d * static_cast<float>(isum[row] - 32 * sum_mins[row]);
    }
  }

  for (size_t row = 0; row < sums.size(); ++row) {
    out[row] = sums[row];
  }
#endif
}

inline void dot_q6_k_x8_q8_k_group_neon(
    const ::emel::kernel::detail::quant::block_q6_kx8 * lhs,
    const ::emel::kernel::detail::quant::block_q8_k * rhs,
    const uint64_t block_count,
    float * out) noexcept {
#if !defined(__aarch64__) || !defined(__ARM_NEON) || !defined(__ARM_FEATURE_DOTPROD)
  (void) lhs;
  (void) rhs;
  (void) block_count;
  std::fill(out,
            out + ::emel::kernel::detail::quant::Q6_K_X8_ROWS,
            0.0f);
#else
  constexpr int col_pairs = 4;
  const uint8x16_t m4b = vdupq_n_u8(0x0fu);
  const uint8x16_t mask_lo = vdupq_n_u8(0x03u);
  const uint8x16_t mask_hi = vdupq_n_u8(0x30u);
  float32x4_t acc_f32[2];
  acc_f32[0] = vdupq_n_f32(0.0f);
  acc_f32[1] = vdupq_n_f32(0.0f);

  for (uint64_t block = 0; block < block_count; ++block) {
    const auto & q6_block = lhs[block];
    const auto & q8_block = rhs[block];
    const float16x4_t q6_d_0_f16 = vreinterpret_f16_u16(vld1_u16(q6_block.d.data()));
    const float16x4_t q6_d_1_f16 = vreinterpret_f16_u16(vld1_u16(q6_block.d.data() + 4));
    const float32x4_t q6_d_0 = vcvt_f32_f16(q6_d_0_f16);
    const float32x4_t q6_d_1 = vcvt_f32_f16(q6_d_1_f16);
    const float32x4_t q8_d = vdupq_n_f32(q8_block.d);
    const float32x4_t sb_scale_0 = vmulq_f32(q6_d_0, q8_d);
    const float32x4_t sb_scale_1 = vmulq_f32(q6_d_1, q8_d);

    int32x2_t acc[col_pairs];
    for (int idx = 0; idx < col_pairs; ++idx) {
      acc[idx] = vdup_n_s32(0);
    }

    int16_t q6_scales[16 * 8];
    for (int scale = 0; scale < 16; ++scale) {
      const int16x8_t scales =
          vmovl_s8(vld1_s8(q6_block.scales.data() + static_cast<size_t>(scale) * 8u));
      vst1q_s16(q6_scales + static_cast<size_t>(scale) * 8u, scales);
    }

    int32x4_t bias_lo = vdupq_n_s32(0);
    int32x4_t bias_hi = vdupq_n_s32(0);
    for (int scale = 0; scale < 16; scale += 4) {
      const int16x4_t bsums_vec = vld1_s16(q8_block.bsums.data() + scale);
      const int16x4_t scales_lo_0 = vld1_s16(q6_scales + (scale + 0) * 8);
      const int16x4_t scales_hi_0 = vld1_s16(q6_scales + (scale + 0) * 8 + 4);
      const int16x4_t scales_lo_1 = vld1_s16(q6_scales + (scale + 1) * 8);
      const int16x4_t scales_hi_1 = vld1_s16(q6_scales + (scale + 1) * 8 + 4);
      const int16x4_t scales_lo_2 = vld1_s16(q6_scales + (scale + 2) * 8);
      const int16x4_t scales_hi_2 = vld1_s16(q6_scales + (scale + 2) * 8 + 4);
      const int16x4_t scales_lo_3 = vld1_s16(q6_scales + (scale + 3) * 8);
      const int16x4_t scales_hi_3 = vld1_s16(q6_scales + (scale + 3) * 8 + 4);

      bias_lo = vmlal_lane_s16(bias_lo, scales_lo_0, bsums_vec, 0);
      bias_hi = vmlal_lane_s16(bias_hi, scales_hi_0, bsums_vec, 0);
      bias_lo = vmlal_lane_s16(bias_lo, scales_lo_1, bsums_vec, 1);
      bias_hi = vmlal_lane_s16(bias_hi, scales_hi_1, bsums_vec, 1);
      bias_lo = vmlal_lane_s16(bias_lo, scales_lo_2, bsums_vec, 2);
      bias_hi = vmlal_lane_s16(bias_hi, scales_hi_2, bsums_vec, 2);
      bias_lo = vmlal_lane_s16(bias_lo, scales_lo_3, bsums_vec, 3);
      bias_hi = vmlal_lane_s16(bias_hi, scales_hi_3, bsums_vec, 3);
    }
    bias_lo = vshlq_n_s32(bias_lo, 5);
    bias_hi = vshlq_n_s32(bias_hi, 5);

    for (int half = 0; half < 2; ++half) {
      const uint8_t * ql_base = q6_block.ql.data() + static_cast<size_t>(half) * 512u;
      const uint8_t * qh_base = q6_block.qh.data() + static_cast<size_t>(half) * 256u;
      for (int sb = 0; sb < static_cast<int>(::emel::kernel::detail::quant::QK_K / 64u); ++sb) {
        const int8_t * q8_base_l = q8_block.qs.data() + static_cast<size_t>(half) * 128u +
            static_cast<size_t>(sb) * 16u;
        const int8_t * q8_base_h = q8_base_l + 64;

        int8x16_t q8_l[2];
        int8x16_t q8_h[2];
        for (int idx = 0; idx < 2; ++idx) {
          const int8x8_t q8_l_half = vld1_s8(q8_base_l + idx * 8);
          const int8x8_t q8_h_half = vld1_s8(q8_base_h + idx * 8);
          q8_l[idx] = vcombine_s8(q8_l_half, q8_l_half);
          q8_h[idx] = vcombine_s8(q8_h_half, q8_h_half);
        }

        const int ql_off_base = sb * static_cast<int>(::emel::kernel::detail::quant::QK_K / 2u);
        const int qh_off_base = ql_off_base & 255;

        uint8x16x4_t q6_ql_0 = vld1q_u8_x4(ql_base + ql_off_base);
        uint8x16x4_t q6_ql_1 = vld1q_u8_x4(ql_base + ql_off_base + 64);
        uint8x16x4_t q6_qh_0 = vld1q_u8_x4(qh_base + qh_off_base);
        uint8x16x4_t q6_qh_1 = vld1q_u8_x4(qh_base + qh_off_base + 64);

        if (sb > 1) {
          q6_qh_0.val[0] = vshrq_n_u8(q6_qh_0.val[0], 2);
          q6_qh_0.val[1] = vshrq_n_u8(q6_qh_0.val[1], 2);
          q6_qh_0.val[2] = vshrq_n_u8(q6_qh_0.val[2], 2);
          q6_qh_0.val[3] = vshrq_n_u8(q6_qh_0.val[3], 2);
          q6_qh_1.val[0] = vshrq_n_u8(q6_qh_1.val[0], 2);
          q6_qh_1.val[1] = vshrq_n_u8(q6_qh_1.val[1], 2);
          q6_qh_1.val[2] = vshrq_n_u8(q6_qh_1.val[2], 2);
          q6_qh_1.val[3] = vshrq_n_u8(q6_qh_1.val[3], 2);
        }

        for (int cp = 0; cp < col_pairs; ++cp) {
          const uint8x16_t q6_qs_cp_0_l = q6_ql_0.val[cp];
          const uint8x16_t q6_qs_cp_1_l = q6_ql_1.val[cp];
          const uint8x16_t q6_qs_cp_0_h = q6_qh_0.val[cp];
          const uint8x16_t q6_qs_cp_1_h = q6_qh_1.val[cp];

          const uint8x16_t q6_qs_cp_0_hh = vandq_u8(q6_qs_cp_0_h, mask_hi);
          const uint8x16_t q6_qs_cp_1_hh = vandq_u8(q6_qs_cp_1_h, mask_hi);

          const int8x16_t q6_l0 = vreinterpretq_s8_u8(
              vsliq_n_u8(vandq_u8(q6_qs_cp_0_l, m4b), vandq_u8(q6_qs_cp_0_h, mask_lo), 4));
          const int8x16_t q6_l1 = vreinterpretq_s8_u8(
              vsliq_n_u8(vandq_u8(q6_qs_cp_1_l, m4b), vandq_u8(q6_qs_cp_1_h, mask_lo), 4));
          const int8x16_t q6_h0 =
              vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_qs_cp_0_l, 4), q6_qs_cp_0_hh));
          const int8x16_t q6_h1 =
              vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6_qs_cp_1_l, 4), q6_qs_cp_1_hh));

          int32x4_t sb_acc_l = vdupq_n_s32(0);
          sb_acc_l = vdotq_s32(sb_acc_l, q6_l0, q8_l[0]);
          sb_acc_l = vdotq_s32(sb_acc_l, q6_l1, q8_l[1]);

          int32x4_t sb_acc_h = vdupq_n_s32(0);
          sb_acc_h = vdotq_s32(sb_acc_h, q6_h0, q8_h[0]);
          sb_acc_h = vdotq_s32(sb_acc_h, q6_h1, q8_h[1]);

          const int32x2_t sum_l = vpadd_s32(vget_low_s32(sb_acc_l), vget_high_s32(sb_acc_l));
          const int32x2_t sum_h = vpadd_s32(vget_low_s32(sb_acc_h), vget_high_s32(sb_acc_h));

          const int scale_idx_l = half * 8 + sb;
          const int scale_idx_h = half * 8 + sb + 4;
          const int32x2_t scale_vec_l = {
              static_cast<int32_t>(q6_scales[static_cast<size_t>(scale_idx_l) * 8u +
                                             static_cast<size_t>(cp) * 2u + 0u]),
              static_cast<int32_t>(q6_scales[static_cast<size_t>(scale_idx_l) * 8u +
                                             static_cast<size_t>(cp) * 2u + 1u]),
          };
          const int32x2_t scale_vec_h = {
              static_cast<int32_t>(q6_scales[static_cast<size_t>(scale_idx_h) * 8u +
                                             static_cast<size_t>(cp) * 2u + 0u]),
              static_cast<int32_t>(q6_scales[static_cast<size_t>(scale_idx_h) * 8u +
                                             static_cast<size_t>(cp) * 2u + 1u]),
          };
          acc[cp] = vmla_s32(acc[cp], sum_l, scale_vec_l);
          acc[cp] = vmla_s32(acc[cp], sum_h, scale_vec_h);
        }
      }
    }

    acc[0] = vsub_s32(acc[0], vget_low_s32(bias_lo));
    acc[1] = vsub_s32(acc[1], vget_high_s32(bias_lo));
    acc[2] = vsub_s32(acc[2], vget_low_s32(bias_hi));
    acc[3] = vsub_s32(acc[3], vget_high_s32(bias_hi));

    const float32x2_t w_01 = vmul_f32(vcvt_f32_s32(acc[0]), vget_low_f32(sb_scale_0));
    const float32x2_t w_23 = vmul_f32(vcvt_f32_s32(acc[1]), vget_high_f32(sb_scale_0));
    const float32x2_t w_45 = vmul_f32(vcvt_f32_s32(acc[2]), vget_low_f32(sb_scale_1));
    const float32x2_t w_67 = vmul_f32(vcvt_f32_s32(acc[3]), vget_high_f32(sb_scale_1));
    acc_f32[0] = vaddq_f32(acc_f32[0], vcombine_f32(w_01, w_23));
    acc_f32[1] = vaddq_f32(acc_f32[1], vcombine_f32(w_45, w_67));
  }

  vst1q_f32(out, acc_f32[0]);
  vst1q_f32(out + 4, acc_f32[1]);
#endif
}

inline void dot_q6_k_x8_q8_k_group_prepared_neon(
    const ::emel::kernel::detail::quant::block_q6_kx8_q8_prepared * lhs,
    const ::emel::kernel::detail::quant::block_q8_k * rhs,
    const uint64_t block_count,
    float * out) noexcept {
#if !defined(__aarch64__) || !defined(__ARM_NEON) || !defined(__ARM_FEATURE_DOTPROD)
  (void) lhs;
  (void) rhs;
  (void) block_count;
  std::fill(
      out, out + ::emel::kernel::detail::quant::Q6_K_X8_ROWS, 0.0f);
#else
  std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> sums = {};
  const int32x4_t zero = vdupq_n_s32(0);
  for (auto & sum : sums) {
    sum = 0.0f;
  }
  for (uint64_t block = 0; block < block_count; ++block) {
    const auto & q6_block = lhs[block];
    const auto & q8_block = rhs[block];
    std::array<int32x4_t, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> acc = {};
    for (auto & row_acc : acc) {
      row_acc = zero;
    }
    const float q8_d = q8_block.d;
    for (uint64_t scale = 0; scale < (::emel::kernel::detail::quant::QK_K / 16u); ++scale) {
      const int8x16_t q8_values = vld1q_s8(
          q8_block.qs.data() + static_cast<size_t>(scale) * 16u);
      for (uint64_t row = 0; row < ::emel::kernel::detail::quant::Q6_K_X8_ROWS; ++row) {
        const size_t block_offset =
            (static_cast<size_t>(scale) * ::emel::kernel::detail::quant::Q6_K_X8_ROWS + row) *
            16u;
        const int8x16_t q6_values = vld1q_s8(q6_block.qs.data() + block_offset);
        const int32x4_t dot_acc = vdotq_s32(zero, q6_values, q8_values);
        const int8_t scale_value = q6_block.scales[static_cast<size_t>(scale) *
                                                       ::emel::kernel::detail::quant::Q6_K_X8_ROWS +
                                                   row];
        acc[row] = vmlaq_n_s32(acc[row], dot_acc, static_cast<int32_t>(scale_value));
      }
    }

    const float16x4_t q6_d_0_f16 = vreinterpret_f16_u16(vld1_u16(q6_block.d.data()));
    const float16x4_t q6_d_1_f16 = vreinterpret_f16_u16(vld1_u16(q6_block.d.data() + 4));
    const float32x4_t block_scale_0 = vmulq_n_f32(vcvt_f32_f16(q6_d_0_f16), q8_d);
    const float32x4_t block_scale_1 = vmulq_n_f32(vcvt_f32_f16(q6_d_1_f16), q8_d);
    const int32x4_t packed_acc_0 = {
        vaddvq_s32(acc[0]),
        vaddvq_s32(acc[1]),
        vaddvq_s32(acc[2]),
        vaddvq_s32(acc[3]),
    };
    const int32x4_t packed_acc_1 = {
        vaddvq_s32(acc[4]),
        vaddvq_s32(acc[5]),
        vaddvq_s32(acc[6]),
        vaddvq_s32(acc[7]),
    };
    const float32x4_t scaled_0 = vmulq_f32(vcvtq_f32_s32(packed_acc_0), block_scale_0);
    const float32x4_t scaled_1 = vmulq_f32(vcvtq_f32_s32(packed_acc_1), block_scale_1);
    float scaled_0_array[4];
    float scaled_1_array[4];
    vst1q_f32(scaled_0_array, scaled_0);
    vst1q_f32(scaled_1_array, scaled_1);
    for (size_t row = 0; row < 4u; ++row) {
      sums[row] += scaled_0_array[row];
      sums[row + 4u] += scaled_1_array[row];
    }
  }

  for (size_t row = 0; row < sums.size(); ++row) {
    out[row] = sums[row];
  }
#endif
}

inline void dot_q6_k_x8_q8_k_group_prepared_i8mm(
    const ::emel::kernel::detail::quant::block_q6_kx8_q8_prepared * lhs,
    const ::emel::kernel::detail::quant::block_q8_k * rhs,
    const uint64_t block_count,
    float * out) noexcept {
#if !defined(__aarch64__) || !defined(__ARM_NEON) || !defined(__ARM_FEATURE_MATMUL_INT8)
  (void) lhs;
  (void) rhs;
  (void) block_count;
  std::fill(out, out + ::emel::kernel::detail::quant::Q6_K_X8_ROWS, 0.0f);
#else
  std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> sums = {};
  const int32x4_t zero = vdupq_n_s32(0);

  for (uint64_t block = 0; block < block_count; ++block) {
    const auto & q6_block = lhs[block];
    const auto & q8_block = rhs[block];
    std::array<int32x4_t, ::emel::kernel::detail::quant::Q6_K_X8_ROWS / 2u> acc_pairs = {};
    for (auto & pair_acc : acc_pairs) {
      pair_acc = zero;
    }

    for (uint64_t scale = 0; scale < (::emel::kernel::detail::quant::QK_K / 16u); ++scale) {
      const int8x16_t q8_values =
          vld1q_s8(q8_block.qs.data() + static_cast<size_t>(scale) * 16u);
      const int8x16_t q8_low_dup = vcombine_s8(vget_low_s8(q8_values), vget_low_s8(q8_values));
      const int8x16_t q8_high_dup =
          vcombine_s8(vget_high_s8(q8_values), vget_high_s8(q8_values));

      for (uint64_t pair = 0; pair < acc_pairs.size(); ++pair) {
        const uint64_t row0 = pair * 2u;
        const uint64_t row1 = row0 + 1u;
        const size_t pair_base =
            static_cast<size_t>(scale) *
                (::emel::kernel::detail::quant::Q6_K_X8_ROWS / 2u) * 32u +
            pair * 32u;
        const int8x16_t lhs_low = vld1q_s8(q6_block.qs.data() + pair_base);
        const int8x16_t lhs_high = vld1q_s8(q6_block.qs.data() + pair_base + 16u);

        int32x4_t pair_dot = vmmlaq_s32(zero, lhs_low, q8_low_dup);
        pair_dot = vmmlaq_s32(pair_dot, lhs_high, q8_high_dup);

        const int8_t scale_row0 =
            q6_block.scales[static_cast<size_t>(scale) *
                                ::emel::kernel::detail::quant::Q6_K_X8_ROWS +
                            row0];
        const int8_t scale_row1 =
            q6_block.scales[static_cast<size_t>(scale) *
                                ::emel::kernel::detail::quant::Q6_K_X8_ROWS +
                            row1];
        const int32x4_t scale_pair = {
            static_cast<int32_t>(scale_row0),
            static_cast<int32_t>(scale_row0),
            static_cast<int32_t>(scale_row1),
            static_cast<int32_t>(scale_row1),
        };
        acc_pairs[pair] = vmlaq_s32(acc_pairs[pair], pair_dot, scale_pair);
      }
    }

    const float q8_d = q8_block.d;
    const float16x4_t q6_d_0_f16 = vreinterpret_f16_u16(vld1_u16(q6_block.d.data()));
    const float16x4_t q6_d_1_f16 = vreinterpret_f16_u16(vld1_u16(q6_block.d.data() + 4));
    const float32x4_t block_scale_0 = vmulq_n_f32(vcvt_f32_f16(q6_d_0_f16), q8_d);
    const float32x4_t block_scale_1 = vmulq_n_f32(vcvt_f32_f16(q6_d_1_f16), q8_d);
    const int32x4_t packed_acc_0 = {
        vgetq_lane_s32(acc_pairs[0], 0),
        vgetq_lane_s32(acc_pairs[0], 2),
        vgetq_lane_s32(acc_pairs[1], 0),
        vgetq_lane_s32(acc_pairs[1], 2),
    };
    const int32x4_t packed_acc_1 = {
        vgetq_lane_s32(acc_pairs[2], 0),
        vgetq_lane_s32(acc_pairs[2], 2),
        vgetq_lane_s32(acc_pairs[3], 0),
        vgetq_lane_s32(acc_pairs[3], 2),
    };
    const float32x4_t scaled_0 = vmulq_f32(vcvtq_f32_s32(packed_acc_0), block_scale_0);
    const float32x4_t scaled_1 = vmulq_f32(vcvtq_f32_s32(packed_acc_1), block_scale_1);
    float scaled_0_array[4];
    float scaled_1_array[4];
    vst1q_f32(scaled_0_array, scaled_0);
    vst1q_f32(scaled_1_array, scaled_1);
    for (size_t row = 0; row < 4u; ++row) {
      sums[row] += scaled_0_array[row];
      sums[row + 4u] += scaled_1_array[row];
    }
  }

  for (size_t row = 0; row < sums.size(); ++row) {
    out[row] = sums[row];
  }
#endif
}

inline void reduce_q6_k_x8_q8_argmax_prepared_i8mm_group(
    const ::emel::kernel::detail::quant::block_q6_kx8_q8_argmax_prepared * lhs,
    const ::emel::kernel::detail::quant::block_q8_k * rhs,
    const uint64_t block_count,
    const uint64_t row_base,
    const uint64_t rows_in_group,
    float & best_value,
    int32_t & best_index,
    bool & have_best) noexcept {
#if !defined(__aarch64__) || !defined(__ARM_NEON) || !defined(__ARM_FEATURE_MATMUL_INT8)
  (void) lhs;
  (void) rhs;
  (void) block_count;
  (void) row_base;
  (void) rows_in_group;
  (void) best_value;
  (void) best_index;
  (void) have_best;
#else
  const int32x4_t zero = vdupq_n_s32(0);
  float32x4_t sums_0 = vdupq_n_f32(0.0f);
  float32x4_t sums_1 = vdupq_n_f32(0.0f);

  for (uint64_t block = 0; block < block_count; ++block) {
    const auto & q6_block = lhs[block];
    const auto & q8_block = rhs[block];
    std::array<int32x4_t, ::emel::kernel::detail::quant::Q6_K_X8_ROWS / 2u> acc_pairs = {};
    for (auto & pair_acc : acc_pairs) {
      pair_acc = zero;
    }

    for (uint64_t scale = 0; scale < (::emel::kernel::detail::quant::QK_K / 16u); ++scale) {
      const int8x16_t q8_values =
          vld1q_s8(q8_block.qs.data() + static_cast<size_t>(scale) * 16u);
      const int8x16_t q8_low_dup = vcombine_s8(vget_low_s8(q8_values), vget_low_s8(q8_values));
      const int8x16_t q8_high_dup =
          vcombine_s8(vget_high_s8(q8_values), vget_high_s8(q8_values));

      for (uint64_t pair = 0; pair < acc_pairs.size(); ++pair) {
        const size_t pair_base =
            static_cast<size_t>(scale) *
                (::emel::kernel::detail::quant::Q6_K_X8_ROWS / 2u) * 32u +
            pair * 32u;
        const int8x16_t lhs_low = vld1q_s8(q6_block.qs.data() + pair_base);
        const int8x16_t lhs_high = vld1q_s8(q6_block.qs.data() + pair_base + 16u);
        int32x4_t pair_dot = vmmlaq_s32(zero, lhs_low, q8_low_dup);
        pair_dot = vmmlaq_s32(pair_dot, lhs_high, q8_high_dup);
        const uint64_t row0 = pair * 2u;
        const uint64_t row1 = row0 + 1u;
        const int8_t scale_row0 =
            q6_block.scales[static_cast<size_t>(scale) *
                                ::emel::kernel::detail::quant::Q6_K_X8_ROWS +
                            row0];
        const int8_t scale_row1 =
            q6_block.scales[static_cast<size_t>(scale) *
                                ::emel::kernel::detail::quant::Q6_K_X8_ROWS +
                            row1];
        const int32x4_t scale_pair = {
            static_cast<int32_t>(scale_row0),
            static_cast<int32_t>(scale_row0),
            static_cast<int32_t>(scale_row1),
            static_cast<int32_t>(scale_row1),
        };
        acc_pairs[pair] = vmlaq_s32(acc_pairs[pair], pair_dot, scale_pair);
      }
    }

    const float32x4_t block_scale_0 = vmulq_n_f32(vld1q_f32(q6_block.d.data()), q8_block.d);
    const float32x4_t block_scale_1 =
        vmulq_n_f32(vld1q_f32(q6_block.d.data() + 4u), q8_block.d);
    const int32x4_t packed_acc_0 = {
        vgetq_lane_s32(acc_pairs[0], 0),
        vgetq_lane_s32(acc_pairs[0], 2),
        vgetq_lane_s32(acc_pairs[1], 0),
        vgetq_lane_s32(acc_pairs[1], 2),
    };
    const int32x4_t packed_acc_1 = {
        vgetq_lane_s32(acc_pairs[2], 0),
        vgetq_lane_s32(acc_pairs[2], 2),
        vgetq_lane_s32(acc_pairs[3], 0),
        vgetq_lane_s32(acc_pairs[3], 2),
    };
    sums_0 = vaddq_f32(sums_0, vmulq_f32(vcvtq_f32_s32(packed_acc_0), block_scale_0));
    sums_1 = vaddq_f32(sums_1, vmulq_f32(vcvtq_f32_s32(packed_acc_1), block_scale_1));
  }

  float group_values[::emel::kernel::detail::quant::Q6_K_X8_ROWS];
  vst1q_f32(group_values, sums_0);
  vst1q_f32(group_values + 4, sums_1);
  for (uint64_t row = 0; row < rows_in_group; ++row) {
    const float value = group_values[row];
    if (!have_best || value > best_value) {
      have_best = true;
      best_value = value;
      best_index = static_cast<int32_t>(row_base + row);
    }
  }
#endif
}

inline void execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_k *>(request.src1.data);
  const uint8_t * prepared = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float * dst = static_cast<float *>(request.dst.data);
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8_q8_prepared *>(
            prepared + group * group_bytes);
    std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> group_out = {};
    dot_q6_k_x8_q8_k_group_prepared_i8mm(group_ptr, q8_blocks, block_count, group_out.data());
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    for (uint64_t row = 0; row < rows_in_group; ++row) {
      dst[row_base + row] = group_out[row];
    }
  }
#endif
}

inline void execute_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm_unchecked(
    const event::op_mul_mat_argmax & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_k *>(request.src1.data);
  const uint8_t * prepared = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float best_value = -std::numeric_limits<float>::infinity();
  int32_t best_index = 0;
  bool have_best = false;
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8_q8_prepared *>(
            prepared + group * group_bytes);
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> group_out = {};
    dot_q6_k_x8_q8_k_group_prepared_i8mm(group_ptr, q8_blocks, block_count, group_out.data());
    for (uint64_t row = 0; row < rows_in_group; ++row) {
      const float value = group_out[row];
      if (!have_best || value > best_value) {
        have_best = true;
        best_value = value;
        best_index = static_cast<int32_t>(row_base + row);
      }
    }
  }

  static_cast<float *>(request.dst.data)[0] = best_value;
  *request.index_out = best_index;
#endif
}

inline void execute_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm_unchecked(
    const event::op_mul_mat_argmax & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_k *>(request.src1.data);
  const uint8_t * prepared = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float best_value = -std::numeric_limits<float>::infinity();
  int32_t best_index = 0;
  bool have_best = false;
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8_q8_argmax_prepared *>(
            prepared + group * group_bytes);
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    reduce_q6_k_x8_q8_argmax_prepared_i8mm_group(
        group_ptr,
        q8_blocks,
        block_count,
        row_base,
        rows_in_group,
        best_value,
        best_index,
        have_best);
  }

  static_cast<float *>(request.dst.data)[0] = best_value;
  *request.index_out = best_index;
#endif
}

inline void execute_neon_mul_mat_argmax_q6_vector_packed_q8_rhs_unchecked(
    const event::op_mul_mat_argmax & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_k *>(request.src1.data);
  const uint8_t * packed = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float best_value = -std::numeric_limits<float>::infinity();
  int32_t best_index = 0;
  bool have_best = false;
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8 *>(
            packed + group * group_bytes);
    std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> group_out = {};
    dot_q6_k_x8_q8_k_group_neon(group_ptr, q8_blocks, block_count, group_out.data());
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    for (uint64_t row = 0; row < rows_in_group; ++row) {
      const float value = group_out[row];
      if (!have_best || value > best_value) {
        have_best = true;
        best_value = value;
        best_index = static_cast<int32_t>(row_base + row);
      }
    }
  }

  static_cast<float *>(request.dst.data)[0] = best_value;
  *request.index_out = best_index;
#endif
}

inline void execute_neon_mul_mat_q6_vector_packed_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  std::array<::emel::kernel::detail::quant::block_q8_k,
             ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
      q8_blocks = {};

  ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
      static_cast<const float *>(request.src1.data),
      1u,
      q8_blocks.data(),
      static_cast<int64_t>(k));

  const uint8_t * packed = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float * dst = static_cast<float *>(request.dst.data);
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8 *>(
            packed + group * group_bytes);
    std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> group_out = {};
    dot_q6_k_x8_q8_k_group_neon(group_ptr, q8_blocks.data(), block_count, group_out.data());
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    for (uint64_t row = 0; row < rows_in_group; ++row) {
      dst[row_base + row] = group_out[row];
    }
  }
#endif
}

inline void execute_neon_mul_mat_q6_vector_packed_q8_rhs_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_k *>(request.src1.data);
  const uint8_t * packed = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float * dst = static_cast<float *>(request.dst.data);
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8 *>(
            packed + group * group_bytes);
    std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> group_out = {};
    dot_q6_k_x8_q8_k_group_neon(group_ptr, q8_blocks, block_count, group_out.data());
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    for (uint64_t row = 0; row < rows_in_group; ++row) {
      dst[row_base + row] = group_out[row];
    }
  }
#endif
}

inline void execute_neon_mul_mat_q6_vector_prepared_q8_rhs_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_k *>(request.src1.data);
  const uint8_t * prepared = static_cast<const uint8_t *>(request.src0.data);
  const size_t group_bytes = request.src0.nb[1];
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q6_k_x8_group_count(m);
  float * dst = static_cast<float *>(request.dst.data);
  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * group_ptr =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_kx8_q8_prepared *>(
            prepared + group * group_bytes);
    std::array<float, ::emel::kernel::detail::quant::Q6_K_X8_ROWS> group_out = {};
    dot_q6_k_x8_q8_k_group_prepared_neon(group_ptr, q8_blocks, block_count, group_out.data());
    const uint64_t row_base = group * ::emel::kernel::detail::quant::Q6_K_X8_ROWS;
    const uint64_t rows_in_group = std::min(
        static_cast<uint64_t>(::emel::kernel::detail::quant::Q6_K_X8_ROWS), m - row_base);
    for (uint64_t row = 0; row < rows_in_group; ++row) {
      dst[row_base + row] = group_out[row];
    }
  }
#endif
}

inline void execute_neon_mul_mat_q6_vector_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  std::array<::emel::kernel::detail::quant::block_q8_k,
             ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
      q8_blocks = {};

  const float * b = static_cast<const float *>(request.src1.data);
  ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
      b,
      1u,
      q8_blocks.data(),
      k);

  const uint8_t * a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  float * c = static_cast<float *>(request.dst.data);

  uint64_t row = 0u;
  for (; row + 4u <= m; row += 4u) {
    const auto * row0 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_k *>(
        a + (row + 0u) * row_bytes);
    const auto * row1 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_k *>(
        a + (row + 1u) * row_bytes);
    const auto * row2 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_k *>(
        a + (row + 2u) * row_bytes);
    const auto * row3 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_k *>(
        a + (row + 3u) * row_bytes);
    float out[4] = {};
    dot_q6_k_q8_k_4rows_neon(row0, row1, row2, row3, q8_blocks.data(), block_count, out);
    c[row + 0u] = out[0];
    c[row + 1u] = out[1];
    c[row + 2u] = out[2];
    c[row + 3u] = out[3];
  }

  for (; row < m; ++row) {
    const auto * row_ptr = reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_k *>(
        a + row * row_bytes);
    c[row] = dot_q6_k_q8_k_row_neon(row_ptr, q8_blocks.data(), block_count);
  }
#endif
}

inline void execute_neon_mul_mat_q8_0_packed_bl4_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(m);
  const auto * packed_base =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0x4 *>(request.src0.data);
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * packed = packed_base + (group * block_count);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (uint64_t block = 0; block < block_count; ++block) {
      const auto & lhs_block = *packed++;
      const auto & rhs_block = q8_blocks[block];
      const int8x16x4_t lhs_low = vld1q_s8_x4(lhs_block.qs.data());
      const int8x16x4_t lhs_high = vld1q_s8_x4(lhs_block.qs.data() + 64);
      const int8x16x2_t rhs_chunks = vld1q_s8_x2(rhs_block.qs.data());
      const float32x4_t lhs_scale =
          vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(lhs_block.d.data())));
      const float32x4_t rhs_scale =
          vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(rhs_block.d)));

      int32x4_t isum = vdupq_n_s32(0);
      isum = vdotq_laneq_s32(isum, lhs_low.val[0], rhs_chunks.val[0], 0);
      isum = vdotq_laneq_s32(isum, lhs_low.val[1], rhs_chunks.val[0], 1);
      isum = vdotq_laneq_s32(isum, lhs_low.val[2], rhs_chunks.val[0], 2);
      isum = vdotq_laneq_s32(isum, lhs_low.val[3], rhs_chunks.val[0], 3);
      isum = vdotq_laneq_s32(isum, lhs_high.val[0], rhs_chunks.val[1], 0);
      isum = vdotq_laneq_s32(isum, lhs_high.val[1], rhs_chunks.val[1], 1);
      isum = vdotq_laneq_s32(isum, lhs_high.val[2], rhs_chunks.val[1], 2);
      isum = vdotq_laneq_s32(isum, lhs_high.val[3], rhs_chunks.val[1], 3);

      acc = vfmaq_f32(acc,
                      vcvtq_f32_s32(isum),
                      vmulq_f32(lhs_scale, rhs_scale));
    }
    store_q8_0_x4_results(dst, group * 4u, m, acc);
  }
#endif
}

inline void execute_neon_mul_mat_q8_0_packed_bl8_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  const auto * packed_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0x4 *>(request.src0.data);
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  for (uint64_t row_base = 0; row_base < m; row_base += 4u) {
    const auto * packed = packed_blocks;
    packed_blocks += block_count;
    const auto * rhs = q8_blocks;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (uint64_t block = 0; block < block_count; ++block) {
      const auto & lhs_block = *packed++;
      const auto & rhs_block = *rhs++;
      const int8x16x4_t lhs_low = vld1q_s8_x4(lhs_block.qs.data());
      const int8x16x4_t lhs_high = vld1q_s8_x4(lhs_block.qs.data() + 64);
      const float16x4_t lhs_scale = vld1_f16(reinterpret_cast<const __fp16 *>(lhs_block.d.data()));
      const int8x8x4_t rhs_chunks = vld1_s8_x4(rhs_block.qs.data());
      const int8x16_t rhs0 = vcombine_s8(rhs_chunks.val[0], rhs_chunks.val[0]);
      const int8x16_t rhs1 = vcombine_s8(rhs_chunks.val[1], rhs_chunks.val[1]);
      const int8x16_t rhs2 = vcombine_s8(rhs_chunks.val[2], rhs_chunks.val[2]);
      const int8x16_t rhs3 = vcombine_s8(rhs_chunks.val[3], rhs_chunks.val[3]);
      const float16x4_t rhs_scale =
          vld1_dup_f16(reinterpret_cast<const __fp16 *>(&rhs_block.d));

      int32x4_t isum0 = vdupq_n_s32(0);
      int32x4_t isum1 = vdupq_n_s32(0);
      isum0 = vdotq_s32(isum0, lhs_low.val[0], rhs0);
      isum1 = vdotq_s32(isum1, lhs_low.val[1], rhs0);
      isum0 = vdotq_s32(isum0, lhs_low.val[2], rhs1);
      isum1 = vdotq_s32(isum1, lhs_low.val[3], rhs1);
      isum0 = vdotq_s32(isum0, lhs_high.val[0], rhs2);
      isum1 = vdotq_s32(isum1, lhs_high.val[1], rhs2);
      isum0 = vdotq_s32(isum0, lhs_high.val[2], rhs3);
      isum1 = vdotq_s32(isum1, lhs_high.val[3], rhs3);
      const int32x4_t isum = vpaddq_s32(isum0, isum1);

      acc = vfmaq_f32(acc,
                      vcvtq_f32_s32(isum),
                      vmulq_f32(vcvt_f32_f16(lhs_scale), vcvt_f32_f16(rhs_scale)));
    }
    store_q8_0_x4_results(dst, row_base, m, acc);
  }
#endif
}

inline void execute_neon_mul_mat_q8_0_packed_bl8_full_groups_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  const uint64_t group_count = ::emel::kernel::detail::quant::packed_q8_0_x4_group_count(m);
  const auto * packed_base =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0x4 *>(request.src0.data);
  const auto * q8_blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  for (uint64_t group = 0; group < group_count; ++group) {
    const auto * packed = packed_base + (group * block_count);
    const auto * rhs = q8_blocks;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (uint64_t block = 0; block < block_count; ++block) {
      const auto & lhs_block = *packed++;
      const auto & rhs_block = *rhs++;
      const int8x16x4_t lhs_low = vld1q_s8_x4(lhs_block.qs.data());
      const int8x16x4_t lhs_high = vld1q_s8_x4(lhs_block.qs.data() + 64);
      const float16x4_t lhs_scale =
          vld1_f16(reinterpret_cast<const __fp16 *>(lhs_block.d.data()));
      const int8x8x4_t rhs_chunks = vld1_s8_x4(rhs_block.qs.data());
      const int8x16_t rhs0 = vcombine_s8(rhs_chunks.val[0], rhs_chunks.val[0]);
      const int8x16_t rhs1 = vcombine_s8(rhs_chunks.val[1], rhs_chunks.val[1]);
      const int8x16_t rhs2 = vcombine_s8(rhs_chunks.val[2], rhs_chunks.val[2]);
      const int8x16_t rhs3 = vcombine_s8(rhs_chunks.val[3], rhs_chunks.val[3]);
      const float16x4_t rhs_scale =
          vld1_dup_f16(reinterpret_cast<const __fp16 *>(&rhs_block.d));

      int32x4_t isum0 = vdupq_n_s32(0);
      int32x4_t isum1 = vdupq_n_s32(0);
      isum0 = vdotq_s32(isum0, lhs_low.val[0], rhs0);
      isum1 = vdotq_s32(isum1, lhs_low.val[1], rhs0);
      isum0 = vdotq_s32(isum0, lhs_low.val[2], rhs1);
      isum1 = vdotq_s32(isum1, lhs_low.val[3], rhs1);
      isum0 = vdotq_s32(isum0, lhs_high.val[0], rhs2);
      isum1 = vdotq_s32(isum1, lhs_high.val[1], rhs2);
      isum0 = vdotq_s32(isum0, lhs_high.val[2], rhs3);
      isum1 = vdotq_s32(isum1, lhs_high.val[3], rhs3);
      const int32x4_t isum = vpaddq_s32(isum0, isum1);

      acc = vfmaq_f32(acc,
                      vcvtq_f32_s32(isum),
                      vmulq_f32(vcvt_f32_f16(lhs_scale), vcvt_f32_f16(rhs_scale)));
    }
    vst1q_f32(dst + (group * ::emel::kernel::detail::quant::Q8_0_X4_ROWS), acc);
  }
#endif
}

inline void execute_neon_mul_mat_q8_0_packed_bl8_matrix_x4_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  const auto * packed_weights_base =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0x4 *>(request.src0.data);
  const auto * packed_rhs_base =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0x4 *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);
  const size_t dst_row_stride = request.dst.nb[0] / sizeof(float);

  for (uint64_t col_base = 0; col_base < m; col_base += 4u) {
    const auto * packed_weights = packed_weights_base + ((col_base / 4u) * block_count);
    const auto * packed_rhs = packed_rhs_base;
    float32x4_t acc_f32[4];

    for (int row = 0; row < 4; ++row) {
      acc_f32[row] = vdupq_n_f32(0.0f);
    }

    for (uint64_t block = 0; block < block_count; ++block) {
      int32x4_t acc[4];

      for (int row = 0; row < 4; ++row) {
        acc[row] = vdupq_n_s32(0);
      }

      for (int chunk = 0; chunk < 4; ++chunk) {
        const int8x16_t rhs01 = vld1q_s8(packed_rhs->qs.data() + chunk * 32);
        const int8x16_t rhs23 = vld1q_s8(packed_rhs->qs.data() + chunk * 32 + 16);
        const int8x16_t lhs01 = vld1q_s8(packed_weights->qs.data() + chunk * 32);
        const int8x16_t lhs23 = vld1q_s8(packed_weights->qs.data() + chunk * 32 + 16);

        acc[0] = vmmlaq_s32(acc[0], rhs01, lhs01);
        acc[1] = vmmlaq_s32(acc[1], rhs01, lhs23);
        acc[2] = vmmlaq_s32(acc[2], rhs23, lhs01);
        acc[3] = vmmlaq_s32(acc[3], rhs23, lhs23);
      }

      const int32x4_t row0 =
          vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
      const int32x4_t row1 =
          vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
      const int32x4_t row2 =
          vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
      const int32x4_t row3 =
          vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));

      const float32x4_t rhs_scale =
          vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16 *>(packed_rhs->d.data())));
      const float32x4_t lhs_scale =
          vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16 *>(packed_weights->d.data())));

      acc_f32[0] = vfmaq_f32(
          acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(lhs_scale, rhs_scale, 0));
      acc_f32[1] = vfmaq_f32(
          acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(lhs_scale, rhs_scale, 1));
      acc_f32[2] = vfmaq_f32(
          acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(lhs_scale, rhs_scale, 2));
      acc_f32[3] = vfmaq_f32(
          acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(lhs_scale, rhs_scale, 3));

      ++packed_rhs;
      ++packed_weights;
    }

    for (int row = 0; row < 4; ++row) {
      vst1q_f32(dst + (static_cast<size_t>(row) * dst_row_stride) + col_base, acc_f32[row]);
    }
  }
#endif
}

inline void execute_neon_mul_mat_q8_0_vector_unchecked(
    const event::op_mul_mat & request) noexcept {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  (void) request;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK8_0;
  alignas(64) std::array<::emel::kernel::detail::quant::block_q8_0,
                         ::emel::kernel::detail::quant::MAX_Q8_0_BLOCKS>
      q8_blocks = {};

  ::emel::kernel::detail::quant::quantize_row_q8_0_strided(
      static_cast<const float *>(request.src1.data),
      1u,
      q8_blocks.data(),
      static_cast<int64_t>(k));

  const uint8_t * a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  float * c = static_cast<float *>(request.dst.data);
  uint64_t row = 0u;
  for (; row + 4u <= m; row += 4u) {
    const auto * row0 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
        a + (row + 0u) * row_bytes);
    const auto * row1 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
        a + (row + 1u) * row_bytes);
    const auto * row2 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
        a + (row + 2u) * row_bytes);
    const auto * row3 = reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
        a + (row + 3u) * row_bytes);
    c[row + 0u] = dot_q8_0_q8_0_row_neon(row0, q8_blocks.data(), block_count);
    c[row + 1u] = dot_q8_0_q8_0_row_neon(row1, q8_blocks.data(), block_count);
    c[row + 2u] = dot_q8_0_q8_0_row_neon(row2, q8_blocks.data(), block_count);
    c[row + 3u] = dot_q8_0_q8_0_row_neon(row3, q8_blocks.data(), block_count);
  }

  for (; row < m; ++row) {
    const auto * row_ptr = reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
        a + row * row_bytes);
    c[row] = dot_q8_0_q8_0_row_neon(row_ptr, q8_blocks.data(), block_count);
  }
#endif
}

inline bool execute_neon_mul_mat(const event::op_mul_mat & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool valid_dims = k != 0 && m != 0 && n != 0;
  const bool valid_layout =
      request.src1.ne[1] == k && request.dst.ne[0] == n && request.dst.ne[1] == m;
  const bool valid = valid_dims && valid_layout;
  const uint64_t valid_u64 = static_cast<uint64_t>(valid);
  const uint8_t src0_type = ::emel::kernel::detail::dtype_code(request.src0.type);
  const float * b = static_cast<const float *>(request.src1.data);
  float * c = static_cast<float *>(request.dst.data);
  const bool quantized_src0 = ::emel::kernel::detail::is_quantized_k_dtype(src0_type);

  if (quantized_src0) {
    const uint8_t * a = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
    std::array<::emel::kernel::detail::quant::block_q8_k,
               ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
        q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }

    for (uint64_t j = 0; j < n * valid_u64; ++j) {
      for (uint64_t block = 0; block < block_count; ++block) {
        ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
            b + block * ::emel::kernel::detail::quant::QK_K * n + j,
            n,
            &q8_blocks[block],
            ::emel::kernel::detail::quant::QK_K);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t * row_ptr = a + i * row_bytes;
        if (src0_type == ::emel::kernel::detail::dtype_q2_k) {
          c[i * n + j] = dot_q2_k_q8_k_row_neon(
              reinterpret_cast<const ::emel::kernel::detail::quant::block_q2_k *>(row_ptr),
              q8_blocks.data(),
              block_count);
        } else if (src0_type == ::emel::kernel::detail::dtype_q3_k) {
          c[i * n + j] = dot_q3_k_q8_k_row_neon(
              reinterpret_cast<const ::emel::kernel::detail::quant::block_q3_k *>(row_ptr),
              q8_blocks.data(),
              block_count);
        } else {
          c[i * n + j] = dot_q6_k_q8_k_row_neon(
              reinterpret_cast<const ::emel::kernel::detail::quant::block_q6_k *>(row_ptr),
              q8_blocks.data(),
              block_count);
        }
      }
    }

    return valid;
  }

  const float * a = static_cast<const float *>(request.src0.data);

  constexpr uint64_t row_block = 4;
  constexpr uint64_t col_vec = 4;
  constexpr uint64_t col_block = 64;
  constexpr uint64_t depth_block = 64;
  alignas(64) static thread_local float packed_b[depth_block * col_block];

  for (uint64_t jb = 0; jb < n * valid_u64; jb += col_block) {
    const uint64_t j_end = std::min<uint64_t>(n, jb + col_block);
    const uint64_t vec_cols = ((j_end - jb) / col_vec) * col_vec;
    const uint64_t j_vec_end = jb + vec_cols;

    for (uint64_t pb = 0; pb < k * valid_u64; pb += depth_block) {
      const uint64_t depth = std::min<uint64_t>(depth_block, k - pb);
      const bool first_depth_block = (pb == 0);
      const float32x4_t zero = vdupq_n_f32(0.0f);
      const uint32x4_t depth_reset_mask =
          vdupq_n_u32(static_cast<uint32_t>(-static_cast<int32_t>(first_depth_block)));

      for (uint64_t kk = 0; kk < depth; ++kk) {
        const float * b_src = b + (pb + kk) * n + jb;
        float * b_dst = packed_b + kk * vec_cols;
        std::memcpy(b_dst, b_src, static_cast<size_t>(vec_cols) * sizeof(float));
#if defined(__GNUC__) || defined(__clang__)
        const uint64_t prefetch_distance =
            16u * static_cast<uint64_t>((kk & 15u) == 0u && kk + 16u < depth);
        __builtin_prefetch(b + (pb + kk + prefetch_distance) * n + jb, 0, 1);
#endif
      }

      for (uint64_t j = jb; j < j_vec_end; j += col_vec) {
        const uint64_t j_offset = j - jb;
        uint64_t i = 0;
        for (; i + row_block <= m; i += row_block) {
          float32x4_t acc0 = vld1q_f32(c + (i + 0) * n + j);
          float32x4_t acc1 = vld1q_f32(c + (i + 1) * n + j);
          float32x4_t acc2 = vld1q_f32(c + (i + 2) * n + j);
          float32x4_t acc3 = vld1q_f32(c + (i + 3) * n + j);
          acc0 = vbslq_f32(depth_reset_mask, zero, acc0);
          acc1 = vbslq_f32(depth_reset_mask, zero, acc1);
          acc2 = vbslq_f32(depth_reset_mask, zero, acc2);
          acc3 = vbslq_f32(depth_reset_mask, zero, acc3);

          for (uint64_t kk = 0; kk < depth; ++kk) {
            const float32x4_t bv = vld1q_f32(packed_b + kk * vec_cols + j_offset);
            acc0 = vmlaq_n_f32(acc0, bv, a[(i + 0) * k + pb + kk]);
            acc1 = vmlaq_n_f32(acc1, bv, a[(i + 1) * k + pb + kk]);
            acc2 = vmlaq_n_f32(acc2, bv, a[(i + 2) * k + pb + kk]);
            acc3 = vmlaq_n_f32(acc3, bv, a[(i + 3) * k + pb + kk]);
          }

          vst1q_f32(c + (i + 0) * n + j, acc0);
          vst1q_f32(c + (i + 1) * n + j, acc1);
          vst1q_f32(c + (i + 2) * n + j, acc2);
          vst1q_f32(c + (i + 3) * n + j, acc3);
        }

        for (; i < m; ++i) {
          float32x4_t acc = vld1q_f32(c + i * n + j);
          acc = vbslq_f32(depth_reset_mask, zero, acc);
          for (uint64_t kk = 0; kk < depth; ++kk) {
            const float32x4_t bv = vld1q_f32(packed_b + kk * vec_cols + j_offset);
            acc = vmlaq_n_f32(acc, bv, a[i * k + pb + kk]);
          }
          vst1q_f32(c + i * n + j, acc);
        }
      }

      const float preserve_existing = static_cast<float>(!first_depth_block);
      for (uint64_t j = j_vec_end; j < j_end; ++j) {
        for (uint64_t i = 0; i < m; ++i) {
          float acc = c[i * n + j] * preserve_existing;
          for (uint64_t kk = 0; kk < depth; ++kk) {
            acc += a[i * k + pb + kk] * b[(pb + kk) * n + j];
          }
          c[i * n + j] = acc;
        }
      }
    }
  }

  return valid;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_unary(const event::op_unary & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint8_t subop_code = static_cast<uint8_t>(request.subop);
  const size_t is_abs =
      static_cast<size_t>(subop_code == static_cast<uint8_t>(event::unary_subop::abs));
  const size_t is_neg =
      static_cast<size_t>(subop_code == static_cast<uint8_t>(event::unary_subop::neg));
  const size_t is_relu =
      static_cast<size_t>(subop_code == static_cast<uint8_t>(event::unary_subop::relu));
  const size_t kernel_index = is_abs * 1u + is_neg * 2u + is_relu * 3u;
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  using unary_kernel_t = void (*)(const float *, float *, uint64_t) noexcept;
  constexpr unary_kernel_t noop_kernel = +[](const float *, float *, uint64_t) noexcept {};
  constexpr std::array<unary_kernel_t, 4> kernels = {
      noop_kernel,
      execute_neon_unary_abs,
      execute_neon_unary_neg,
      execute_neon_unary_relu,
  };
  kernels[kernel_index](src, dst, count);
  return kernel_index != 0u;
#else
  (void) request;
  return false;
#endif
}

inline void execute_neon_unary_abs_request(const event::op_unary & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  execute_neon_unary_abs(src, dst, count);
#else
  (void) request;
#endif
}

inline void execute_neon_unary_neg_request(const event::op_unary & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  execute_neon_unary_neg(src, dst, count);
#else
  (void) request;
#endif
}

inline void execute_neon_unary_relu_request(const event::op_unary & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  execute_neon_unary_relu(src, dst, count);
#else
  (void) request;
#endif
}

template <event::unary_subop subop>
inline void execute_simd_unary_subop_unchecked(const event::op_unary & request) noexcept {
  if constexpr (subop == event::unary_subop::abs) {
    execute_neon_unary_abs_request(request);
  }
  if constexpr (subop == event::unary_subop::neg) {
    execute_neon_unary_neg_request(request);
  }
  if constexpr (subop == event::unary_subop::relu) {
    execute_neon_unary_relu_request(request);
  }
}

template <class request_type>
inline void execute_simd_unchecked(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    (void) execute_neon_dup(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_add>) {
    (void) execute_neon_add(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sub>) {
    (void) execute_neon_sub(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void) execute_neon_mul(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_div>) {
    (void) execute_neon_div(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    (void) execute_neon_sqr(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    (void) execute_neon_sqrt(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    (void) execute_neon_mul_mat(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    (void) execute_neon_unary(request);
  }
}

template <class request_type>
inline bool execute_simd(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    return execute_neon_dup(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_add>) {
    return execute_neon_add(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sub>) {
    return execute_neon_sub(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    return execute_neon_mul(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_div>) {
    return execute_neon_div(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    return execute_neon_sqr(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    return execute_neon_sqrt(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    return execute_neon_mul_mat(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return execute_neon_unary(request);
  }
  return false;
}

template <class request_type, class context_type>
inline bool execute_request(const request_type & request, const context_type & ctx) noexcept {
  const bool simd_succeeded = can_use_neon(request, ctx.neon_available) && execute_simd(request);
  return simd_succeeded || ::emel::kernel::detail::execute_scalar(request);
}

}  // namespace emel::kernel::aarch64::detail
namespace emel::kernel::aarch64::action {

namespace detail {

template <class dispatch_event_type>
inline void mark_done(const dispatch_event_type & ev, context & ctx) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::done;
  ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::none));
}

template <class dispatch_event_type>
inline void mark_error(const dispatch_event_type & ev, context & ctx,
                       const int32_t err) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::failed;
  ev.ctx.err = err;
}

struct mark_done_op {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    mark_done(ev, ctx);
  }
};

struct exec_dispatch {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_request & ev,
                  context & ctx) const noexcept {
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type>
struct exec_scalar_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    using request_type = std::remove_cvref_t<decltype(ev.request)>;
    if constexpr (std::is_same_v<request_type, ::emel::kernel::event::op_flash_attn_ext>) {
      ::emel::kernel::detail::run_flash_attn_ext_active_kv_with_workspace_unchecked(
          ev.request, ctx.flash_attn_workspace);
      ++ctx.shared_flash_dispatch_count;
      detail::mark_done(ev, ctx);
    } else {
      if constexpr (std::is_same_v<request_type, ::emel::kernel::event::op_mul_mat>) {
        const uint8_t src0_type = ::emel::kernel::detail::dtype_code(ev.request.src0.type);
        ctx.shared_q8_0_dispatch_count +=
            static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q8_0);
        ctx.shared_q2_dispatch_count +=
            static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q2_k);
        ctx.shared_q3_dispatch_count +=
            static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q3_k);
        ctx.shared_q6_dispatch_count +=
            static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q6_k);
      }
      ::emel::kernel::detail::execute_scalar_unchecked(ev.request);
      detail::mark_done(ev, ctx);
    }
  }
};

template <class dispatch_event_type>
struct exec_simd_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    using request_type = std::remove_cvref_t<decltype(ev.request)>;
    if constexpr (std::is_same_v<request_type, ::emel::kernel::event::op_mul_mat>) {
      const uint8_t src0_type = ::emel::kernel::detail::dtype_code(ev.request.src0.type);
      ctx.optimized_q2_dispatch_count +=
          static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q2_k);
      ctx.optimized_q3_dispatch_count +=
          static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q3_k);
      ctx.optimized_q6_dispatch_count +=
          static_cast<uint64_t>(src0_type == ::emel::kernel::detail::dtype_q6_k);
    }
    ::emel::kernel::aarch64::detail::execute_simd_unchecked(ev.request);
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q6_vector_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q8_0_vector_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q8_0_vector_unchecked(ev.request);
    ++ctx.optimized_q8_0_dispatch_count;
    ++ctx.optimized_q8_0_vector_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q8_0_packed_bl4_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q8_0_packed_bl4_unchecked(ev.request);
    ++ctx.optimized_q8_0_dispatch_count;
    ++ctx.optimized_q8_0_packed_dispatch_count;
    ++ctx.optimized_q8_0_packed_bl4_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q8_0_packed_bl8_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q8_0_packed_bl8_unchecked(ev.request);
    ++ctx.optimized_q8_0_dispatch_count;
    ++ctx.optimized_q8_0_packed_dispatch_count;
    ++ctx.optimized_q8_0_packed_bl8_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q8_0_packed_bl8_full_groups_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::
        execute_neon_mul_mat_q8_0_packed_bl8_full_groups_unchecked(ev.request);
    ++ctx.optimized_q8_0_dispatch_count;
    ++ctx.optimized_q8_0_packed_dispatch_count;
    ++ctx.optimized_q8_0_packed_bl8_dispatch_count;
    ++ctx.optimized_q8_0_packed_bl8_full_groups_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q8_0_packed_bl8_matrix_x4_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::
        execute_neon_mul_mat_q8_0_packed_bl8_matrix_x4_unchecked(ev.request);
    ++ctx.optimized_q8_0_dispatch_count;
    ++ctx.optimized_q8_0_packed_dispatch_count;
    ++ctx.optimized_q8_0_packed_bl8_dispatch_count;
    ++ctx.optimized_q8_0_packed_bl8_matrix_x4_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_packed_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q6_vector_packed_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_packed_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_packed_q8_rhs_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q6_vector_packed_q8_rhs_unchecked(
        ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_packed_dispatch_count;
    ++ctx.optimized_q6_vector_packed_q8_rhs_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_prepared_q8_rhs_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q6_vector_prepared_q8_rhs_unchecked(
        ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_packed_dispatch_count;
    ++ctx.optimized_q6_vector_packed_q8_rhs_dispatch_count;
    ++ctx.optimized_q6_vector_prepared_q8_rhs_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_prepared_q8_rhs_i8mm_op_mul_mat {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::
        execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_packed_dispatch_count;
    ++ctx.optimized_q6_vector_packed_q8_rhs_dispatch_count;
    ++ctx.optimized_q6_vector_prepared_q8_rhs_dispatch_count;
    ++ctx.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_prepared_q8_rhs_i8mm_op_mul_mat_argmax {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::
        execute_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_argmax_dispatch_count;
    ++ctx.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_q8_argmax_prepared_i8mm_op_mul_mat_argmax {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::
        execute_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_argmax_dispatch_count;
    ++ctx.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_q6_vector_packed_q8_rhs_op_mul_mat_argmax {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::
        execute_neon_mul_mat_argmax_q6_vector_packed_q8_rhs_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    ++ctx.optimized_q6_vector_dispatch_count;
    ++ctx.optimized_q6_vector_argmax_dispatch_count;
    ++ctx.optimized_q6_vector_packed_dispatch_count;
    ++ctx.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct exec_simd_flash_attn_ext_f16kv_one_chunk {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::run_flash_attn_ext_f16kv_one_chunk_neon_unchecked(
        ev.request, ctx.flash_attn_workspace);
    ++ctx.optimized_flash_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop>
struct exec_simd_unary_op {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_op_unary & ev,
                  context & ctx) const noexcept {
    ::emel::kernel::aarch64::detail::execute_simd_unary_subop_unchecked<subop>(ev.request);
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type>
struct reject_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    detail::mark_error(ev, ctx, static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

}  // namespace detail

using exec_dispatch_t = detail::exec_dispatch;

#define EMEL_KERNEL_DECLARE_RUN_TYPE(op_name)                                \
  using exec_##op_name##_t =                                                  \
      detail::exec_scalar_op<::emel::kernel::aarch64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_RUN_TYPE)
#undef EMEL_KERNEL_DECLARE_RUN_TYPE

using exec_simd_op_dup_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_dup>;
using exec_simd_op_add_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_add>;
using exec_simd_op_sub_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_sub>;
using exec_simd_op_mul_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_mul>;
using exec_simd_op_div_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_div>;
using exec_simd_op_sqr_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_sqr>;
using exec_simd_op_sqrt_t = detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_sqrt>;
using exec_simd_op_mul_mat_q8_0_packed_bl4_t = detail::exec_simd_q8_0_packed_bl4_op_mul_mat;
using exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups_t =
    detail::exec_simd_q8_0_packed_bl8_full_groups_op_mul_mat;
using exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4_t =
    detail::exec_simd_q8_0_packed_bl8_matrix_x4_op_mul_mat;
using exec_simd_op_mul_mat_q8_0_packed_bl8_t = detail::exec_simd_q8_0_packed_bl8_op_mul_mat;
using exec_simd_op_mul_mat_q8_0_vector_t = detail::exec_simd_q8_0_vector_op_mul_mat;
using exec_simd_op_mul_mat_q6_vector_packed_t = detail::exec_simd_q6_vector_packed_op_mul_mat;
using exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_t =
    detail::exec_simd_q6_vector_prepared_q8_rhs_i8mm_op_mul_mat;
using exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs_t =
    detail::exec_simd_q6_vector_packed_q8_rhs_op_mul_mat_argmax;
using exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm_t =
    detail::exec_simd_q6_vector_prepared_q8_rhs_i8mm_op_mul_mat_argmax;
using exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm_t =
    detail::exec_simd_q6_vector_q8_argmax_prepared_i8mm_op_mul_mat_argmax;
using exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_t =
    detail::exec_simd_q6_vector_prepared_q8_rhs_op_mul_mat;
using exec_simd_op_mul_mat_q6_vector_packed_q8_rhs_t =
    detail::exec_simd_q6_vector_packed_q8_rhs_op_mul_mat;
using exec_simd_op_mul_mat_q6_vector_t = detail::exec_simd_q6_vector_op_mul_mat;
using exec_simd_op_flash_attn_ext_f16kv_one_chunk_t =
    detail::exec_simd_flash_attn_ext_f16kv_one_chunk;
using exec_simd_op_mul_mat_t =
    detail::exec_simd_op<::emel::kernel::aarch64::event::dispatch_op_mul_mat>;
using exec_simd_op_unary_abs_t =
    detail::exec_simd_unary_op<::emel::kernel::event::unary_subop::abs>;
using exec_simd_op_unary_neg_t =
    detail::exec_simd_unary_op<::emel::kernel::event::unary_subop::neg>;
using exec_simd_op_unary_relu_t =
    detail::exec_simd_unary_op<::emel::kernel::event::unary_subop::relu>;
using exec_scalar_op_unary_abs_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::aarch64::event::dispatch_op_unary, context, detail::mark_done_op,
    ::emel::kernel::event::unary_subop::abs>;
using exec_scalar_op_unary_neg_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::aarch64::event::dispatch_op_unary, context, detail::mark_done_op,
    ::emel::kernel::event::unary_subop::neg>;
using exec_scalar_op_unary_relu_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::aarch64::event::dispatch_op_unary, context, detail::mark_done_op,
    ::emel::kernel::event::unary_subop::relu>;
using exec_scalar_op_unary_exp_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::aarch64::event::dispatch_op_unary, context, detail::mark_done_op,
    ::emel::kernel::event::unary_subop::exp>;

#define EMEL_KERNEL_DECLARE_REJECT_TYPE(op_name)                                      \
  using reject_invalid_##op_name##_t =                                                \
      detail::reject_op<::emel::kernel::aarch64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_REJECT_TYPE)
#undef EMEL_KERNEL_DECLARE_REJECT_TYPE

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.ctx; }) {
      detail::mark_error(ev, ctx, static_cast<int32_t>(emel::error::cast(error::internal_error)));
    } else {
      ++ctx.dispatch_generation;
    }
  }
};

inline constexpr exec_dispatch_t exec_dispatch{};
inline constexpr exec_simd_op_dup_t exec_simd_op_dup{};
inline constexpr exec_simd_op_add_t exec_simd_op_add{};
inline constexpr exec_simd_op_sub_t exec_simd_op_sub{};
inline constexpr exec_simd_op_mul_t exec_simd_op_mul{};
inline constexpr exec_simd_op_div_t exec_simd_op_div{};
inline constexpr exec_simd_op_sqr_t exec_simd_op_sqr{};
inline constexpr exec_simd_op_sqrt_t exec_simd_op_sqrt{};
inline constexpr exec_simd_op_mul_mat_q8_0_packed_bl4_t exec_simd_op_mul_mat_q8_0_packed_bl4{};
inline constexpr exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups_t
    exec_simd_op_mul_mat_q8_0_packed_bl8_full_groups{};
inline constexpr exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4_t
    exec_simd_op_mul_mat_q8_0_packed_bl8_matrix_x4{};
inline constexpr exec_simd_op_mul_mat_q8_0_packed_bl8_t exec_simd_op_mul_mat_q8_0_packed_bl8{};
inline constexpr exec_simd_op_mul_mat_q8_0_vector_t exec_simd_op_mul_mat_q8_0_vector{};
inline constexpr exec_simd_op_mul_mat_q6_vector_packed_t exec_simd_op_mul_mat_q6_vector_packed{};
inline constexpr exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm_t
    exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm{};
inline constexpr exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs_t
    exec_simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs{};
inline constexpr exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm_t
    exec_simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm{};
inline constexpr exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm_t
    exec_simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm{};
inline constexpr exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs_t
    exec_simd_op_mul_mat_q6_vector_prepared_q8_rhs{};
inline constexpr exec_simd_op_mul_mat_q6_vector_packed_q8_rhs_t
    exec_simd_op_mul_mat_q6_vector_packed_q8_rhs{};
inline constexpr exec_simd_op_mul_mat_q6_vector_t exec_simd_op_mul_mat_q6_vector{};
inline constexpr exec_simd_op_flash_attn_ext_f16kv_one_chunk_t
    exec_simd_op_flash_attn_ext_f16kv_one_chunk{};
inline constexpr exec_simd_op_mul_mat_t exec_simd_op_mul_mat{};
inline constexpr exec_simd_op_unary_abs_t exec_simd_op_unary_abs{};
inline constexpr exec_simd_op_unary_neg_t exec_simd_op_unary_neg{};
inline constexpr exec_simd_op_unary_relu_t exec_simd_op_unary_relu{};
inline constexpr exec_scalar_op_unary_abs_t exec_scalar_op_unary_abs{};
inline constexpr exec_scalar_op_unary_neg_t exec_scalar_op_unary_neg{};
inline constexpr exec_scalar_op_unary_relu_t exec_scalar_op_unary_relu{};
inline constexpr exec_scalar_op_unary_exp_t exec_scalar_op_unary_exp{};

#define EMEL_KERNEL_DEFINE_RUN_ACTION(op_name) \
  inline constexpr exec_##op_name##_t exec_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_RUN_ACTION)
#undef EMEL_KERNEL_DEFINE_RUN_ACTION

#define EMEL_KERNEL_DEFINE_REJECT_ACTION(op_name)            \
  inline constexpr reject_invalid_##op_name##_t reject_invalid_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_REJECT_ACTION)
#undef EMEL_KERNEL_DEFINE_REJECT_ACTION

inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::aarch64::action
