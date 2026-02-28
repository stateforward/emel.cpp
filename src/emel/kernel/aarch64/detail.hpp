#pragma once

#include <cmath>
#include <cstdint>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::aarch64::detail {

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

  if (!neon_available) {
    return false;
  }
  if (!::emel::kernel::detail::can_execute_scalar(request)) {
    return false;
  }
  if (::emel::kernel::detail::dtype_code(request.src0.type) !=
          ::emel::kernel::detail::dtype_f32 ||
      ::emel::kernel::detail::dtype_code(request.dst.type) !=
          ::emel::kernel::detail::dtype_f32) {
    return false;
  }

  if constexpr (::emel::kernel::detail::requires_src1_v<request_type>) {
    if (::emel::kernel::detail::dtype_code(request.src1.type) !=
        ::emel::kernel::detail::dtype_f32) {
      return false;
    }
    if (!is_dense_contiguous(request.src1)) {
      return false;
    }
  }

  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    if (!unary_subop_supported_simd(request.subop)) {
      return false;
    }
  }

  return is_dense_contiguous(request.src0) && is_dense_contiguous(request.dst);
#endif
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

inline bool execute_neon_mul_mat(const event::op_mul_mat & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  if (k == 0 || m == 0 || n == 0) {
    return false;
  }
  if (request.src1.ne[1] != k || request.dst.ne[0] != n || request.dst.ne[1] != m) {
    return false;
  }

  const float * a = static_cast<const float *>(request.src0.data);
  const float * b = static_cast<const float *>(request.src1.data);
  float * c = static_cast<float *>(request.dst.data);

  for (uint64_t i = 0; i < m; ++i) {
    uint64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (uint64_t p = 0; p < k; ++p) {
        const float32x4_t bv = vld1q_f32(b + p * n + j);
        acc = vmlaq_n_f32(acc, bv, a[i * k + p]);
      }
      vst1q_f32(c + i * n + j, acc);
    }

    for (; j < n; ++j) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < k; ++p) {
        acc += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = acc;
    }
  }

  return true;
#else
  (void) request;
  return false;
#endif
}

inline bool execute_neon_unary(const event::op_unary & request) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  if (!unary_subop_supported_simd(request.subop)) {
    return false;
  }

  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  const auto subop_code = static_cast<uint8_t>(request.subop);

  uint64_t i = 0;
  const float32x4_t zero = vdupq_n_f32(0.0f);
  if (subop_code == static_cast<uint8_t>(event::unary_subop::abs)) {
    for (; i + 4 <= count; i += 4) {
      const float32x4_t v = vld1q_f32(src + i);
      vst1q_f32(dst + i, vabsq_f32(v));
    }
    for (; i < count; ++i) {
      dst[i] = std::fabs(src[i]);
    }
    return true;
  }

  if (subop_code == static_cast<uint8_t>(event::unary_subop::neg)) {
    for (; i + 4 <= count; i += 4) {
      const float32x4_t v = vld1q_f32(src + i);
      vst1q_f32(dst + i, vnegq_f32(v));
    }
    for (; i < count; ++i) {
      dst[i] = -src[i];
    }
    return true;
  }

  if (subop_code == static_cast<uint8_t>(event::unary_subop::relu)) {
    for (; i + 4 <= count; i += 4) {
      const float32x4_t v = vld1q_f32(src + i);
      vst1q_f32(dst + i, vmaxq_f32(v, zero));
    }
    for (; i < count; ++i) {
      dst[i] = std::max(0.0f, src[i]);
    }
    return true;
  }

  return false;
#else
  (void) request;
  return false;
#endif
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
  if (can_use_neon(request, ctx.neon_available) && execute_simd(request)) {
    return true;
  }
  return ::emel::kernel::detail::execute_scalar(request);
}

}  // namespace emel::kernel::aarch64::detail
