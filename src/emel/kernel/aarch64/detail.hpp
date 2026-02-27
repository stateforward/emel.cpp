#pragma once

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
  const uint8_t code = ::emel::kernel::detail::dtype_code(tensor.type);
  const uint64_t elem_size = ::emel::kernel::detail::dtype_size_bytes(code);
  if (elem_size == 0) {
    return false;
  }

  if (tensor.nb[0] == 0) {
    return true;
  }

  uint64_t stride = elem_size;
  for (size_t i = 0; i < 4; ++i) {
    if (tensor.nb[i] != stride) {
      return false;
    }
    stride *= tensor.ne[i];
  }
  return true;
}

template <class request_type>
inline bool can_use_neon(const request_type & request, const bool neon_available) noexcept {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  (void) request;
  (void) neon_available;
  return false;
#else
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

template <class request_type>
inline bool execute_simd(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    return execute_neon_dup(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_add>) {
    return execute_neon_add(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    return execute_neon_mul(request);
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
