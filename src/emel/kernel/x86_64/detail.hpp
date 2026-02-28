#pragma once

#include <algorithm>
#include <cmath>
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
inline bool can_use_avx2(const request_type & request, const bool avx2_available) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void) request;
  (void) avx2_available;
  return false;
#else
  if constexpr (!simd_supported_request_v<request_type>) {
    return false;
  }

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

  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    if (!unary_subop_supported_simd(request.subop)) {
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
inline bool execute_avx2_sub(const event::op_sub & request) noexcept {
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
    _mm256_storeu_ps(dst + i, _mm256_sub_ps(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] - rhs[i];
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

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_div(const event::op_div & request) noexcept {
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
    _mm256_storeu_ps(dst + i, _mm256_div_ps(a, b));
  }
  for (; i < count; ++i) {
    dst[i] = lhs[i] / rhs[i];
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
inline bool execute_avx2_sqr(const event::op_sqr & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, v));
  }
  for (; i < count; ++i) {
    dst[i] = src[i] * src[i];
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
inline bool execute_avx2_sqrt(const event::op_sqrt & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);

  uint64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_sqrt_ps(v));
  }
  for (; i < count; ++i) {
    dst[i] = std::sqrt(src[i]);
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
inline bool execute_avx2_mul_mat(const event::op_mul_mat & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
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

  constexpr uint64_t row_block = 4;
  constexpr uint64_t col_vec = 8;
  constexpr uint64_t col_block = 64;

  uint64_t i = 0;
  for (; i + row_block <= m; i += row_block) {
    for (uint64_t jb = 0; jb < n; jb += col_block) {
      const uint64_t j_end = std::min<uint64_t>(n, jb + col_block);
      uint64_t j = jb;
      for (; j + col_vec <= j_end; j += col_vec) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (uint64_t p = 0; p < k; ++p) {
          const __m256 bv = _mm256_loadu_ps(b + p * n + j);

          const __m256 a0 = _mm256_set1_ps(a[(i + 0) * k + p]);
          const __m256 a1 = _mm256_set1_ps(a[(i + 1) * k + p]);
          const __m256 a2 = _mm256_set1_ps(a[(i + 2) * k + p]);
          const __m256 a3 = _mm256_set1_ps(a[(i + 3) * k + p]);

          acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, bv));
          acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, bv));
          acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(a2, bv));
          acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(a3, bv));
#if defined(__GNUC__) || defined(__clang__)
          if ((p & 15u) == 0 && p + 16u < k) {
            _mm_prefetch(reinterpret_cast<const char *>(b + (p + 16u) * n + j), _MM_HINT_T0);
          }
#endif
        }

        _mm256_storeu_ps(c + (i + 0) * n + j, acc0);
        _mm256_storeu_ps(c + (i + 1) * n + j, acc1);
        _mm256_storeu_ps(c + (i + 2) * n + j, acc2);
        _mm256_storeu_ps(c + (i + 3) * n + j, acc3);
      }

      for (; j < j_end; ++j) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (uint64_t p = 0; p < k; ++p) {
          const float bv = b[p * n + j];
          acc0 += a[(i + 0) * k + p] * bv;
          acc1 += a[(i + 1) * k + p] * bv;
          acc2 += a[(i + 2) * k + p] * bv;
          acc3 += a[(i + 3) * k + p] * bv;
        }
        c[(i + 0) * n + j] = acc0;
        c[(i + 1) * n + j] = acc1;
        c[(i + 2) * n + j] = acc2;
        c[(i + 3) * n + j] = acc3;
      }
    }
  }

  for (; i < m; ++i) {
    for (uint64_t jb = 0; jb < n; jb += col_block) {
      const uint64_t j_end = std::min<uint64_t>(n, jb + col_block);
      uint64_t j = jb;
      for (; j + col_vec <= j_end; j += col_vec) {
        __m256 acc = _mm256_setzero_ps();
        for (uint64_t p = 0; p < k; ++p) {
          const __m256 bv = _mm256_loadu_ps(b + p * n + j);
          const __m256 av = _mm256_set1_ps(a[i * k + p]);
          acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
#if defined(__GNUC__) || defined(__clang__)
          if ((p & 15u) == 0 && p + 16u < k) {
            _mm_prefetch(reinterpret_cast<const char *>(b + (p + 16u) * n + j), _MM_HINT_T0);
          }
#endif
        }
        _mm256_storeu_ps(c + i * n + j, acc);
      }

      for (; j < j_end; ++j) {
        float acc = 0.0f;
        for (uint64_t p = 0; p < k; ++p) {
          acc += a[i * k + p] * b[p * n + j];
        }
        c[i * n + j] = acc;
      }
    }
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
inline bool execute_avx2_unary(const event::op_unary & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  if (!unary_subop_supported_simd(request.subop)) {
    return false;
  }

  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  const auto subop_code = static_cast<uint8_t>(request.subop);

  uint64_t i = 0;
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 zero = _mm256_setzero_ps();

  if (subop_code == static_cast<uint8_t>(event::unary_subop::abs)) {
    for (; i + 8 <= count; i += 8) {
      const __m256 v = _mm256_loadu_ps(src + i);
      _mm256_storeu_ps(dst + i, _mm256_andnot_ps(sign_mask, v));
    }
    for (; i < count; ++i) {
      dst[i] = std::fabs(src[i]);
    }
    return true;
  }

  if (subop_code == static_cast<uint8_t>(event::unary_subop::neg)) {
    for (; i + 8 <= count; i += 8) {
      const __m256 v = _mm256_loadu_ps(src + i);
      _mm256_storeu_ps(dst + i, _mm256_sub_ps(zero, v));
    }
    for (; i < count; ++i) {
      dst[i] = -src[i];
    }
    return true;
  }

  if (subop_code == static_cast<uint8_t>(event::unary_subop::relu)) {
    for (; i + 8 <= count; i += 8) {
      const __m256 v = _mm256_loadu_ps(src + i);
      _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zero));
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
  if constexpr (std::is_same_v<request_type, event::op_sub>) {
    (void) execute_avx2_sub(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void) execute_avx2_mul(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_div>) {
    (void) execute_avx2_div(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    (void) execute_avx2_sqr(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    (void) execute_avx2_sqrt(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    (void) execute_avx2_mul_mat(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    (void) execute_avx2_unary(request);
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
  if constexpr (std::is_same_v<request_type, event::op_sub>) {
    return execute_avx2_sub(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    return execute_avx2_mul(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_div>) {
    return execute_avx2_div(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    return execute_avx2_sqr(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    return execute_avx2_sqrt(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    return execute_avx2_mul_mat(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return execute_avx2_unary(request);
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
