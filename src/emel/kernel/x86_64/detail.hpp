#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define EMEL_KERNEL_X86_AVX2_TARGET __attribute__((target("avx2")))
#else
#define EMEL_KERNEL_X86_AVX2_TARGET
#endif

namespace emel::kernel::x86_64::detail {

inline constexpr bool avx2_intrinsics_compiled =
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__AVX2__) || defined(__GNUC__) || defined(__clang__))
    true;
#else
    false;
#endif

inline bool detect_avx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
#else
  return false;
#endif
}

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type & tensor) noexcept {
  return ::emel::kernel::detail::is_dense_contiguous(tensor);
}

template <class request_type>
inline bool can_use_avx2(const request_type & request, const bool avx2_available) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void) request;
  (void) avx2_available;
  return false;
#else
  if (!avx2_available) {
    return false;
  }
  if (!avx2_intrinsics_compiled) {
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

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_dup(const event::op_dup & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  uint64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, v);
  }
  for (; i < count; ++i) {
    dst[i] = src[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
#else
  (void) request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_add(const event::op_add & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * lhs = static_cast<const float *>(request.src0.data);
  const float * rhs = static_cast<const float *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 a = _mm256_loadu_ps(lhs + i);
    const __m256 b = _mm256_loadu_ps(rhs + i);
    _mm256_storeu_ps(dst + i, _mm256_add_ps(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] + rhs[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
#else
  (void) request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_mul(const event::op_mul & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * lhs = static_cast<const float *>(request.src0.data);
  const float * rhs = static_cast<const float *>(request.src1.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 a = _mm256_loadu_ps(lhs + i);
    const __m256 b = _mm256_loadu_ps(rhs + i);
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] * rhs[i];
  }
  return true;
#else
  (void) request;
  return false;
#endif
#else
  (void) request;
  return false;
#endif
}

template <class request_type>
inline void execute_simd_unchecked(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    (void) execute_avx2_dup(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_add>) {
    (void) execute_avx2_add(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void) execute_avx2_mul(request);
  }
}

template <class request_type>
inline bool execute_simd(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    return execute_avx2_dup(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_add>) {
    return execute_avx2_add(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    return execute_avx2_mul(request);
  }
  return false;
}

template <class request_type, class context_type>
inline bool execute_request(const request_type & request, const context_type & ctx) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  if (can_use_avx2(request, ctx.avx2_available) && execute_simd(request)) {
    return true;
  }
#else
  (void) ctx;
#endif
  return ::emel::kernel::detail::execute_scalar(request);
}

}  // namespace emel::kernel::x86_64::detail

#undef EMEL_KERNEL_X86_AVX2_TARGET
