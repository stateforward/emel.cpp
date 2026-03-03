#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/emel.h"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/errors.hpp"
#include "emel/kernel/x86_64/events.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_X86_AVX2_TARGET __attribute__((target("avx2")))
#else
#define EMEL_KERNEL_X86_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_X86_AVX2_TARGET
#endif

namespace emel::kernel::x86_64::detail {

namespace event = ::emel::kernel::event;

inline constexpr bool avx2_intrinsics_compiled =
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
    true;
#else
    false;
#endif
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

EMEL_KERNEL_X86_AVX2_TARGET
inline void execute_avx2_unary_abs(const float * src, float * dst, const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  uint64_t i = 0;
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_andnot_ps(sign_mask, v));
  }
  for (; i < count; ++i) {
    dst[i] = std::fabs(src[i]);
  }
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void execute_avx2_unary_neg(const float * src, float * dst, const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  uint64_t i = 0;
  const __m256 zero = _mm256_setzero_ps();
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_sub_ps(zero, v));
  }
  for (; i < count; ++i) {
    dst[i] = -src[i];
  }
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void execute_avx2_unary_relu(const float * src, float * dst, const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  uint64_t i = 0;
  const __m256 zero = _mm256_setzero_ps();
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zero));
  }
  for (; i < count; ++i) {
    dst[i] = std::max(0.0f, src[i]);
  }
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
#else
  (void) src;
  (void) dst;
  (void) count;
#endif
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

  const bool base_supported =
      avx2_available &&
      avx2_intrinsics_compiled &&
      ::emel::kernel::detail::can_execute_scalar(request) &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32;
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
      is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.dst);
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
  const bool valid_dims = k != 0 && m != 0 && n != 0;
  const bool valid_layout =
      request.src1.ne[1] == k && request.dst.ne[0] == n && request.dst.ne[1] == m;
  {
    const size_t emel_branch_valid = static_cast<size_t>(valid_dims && valid_layout);
    for (size_t emel_case_valid = emel_branch_valid; emel_case_valid == 0u;
         emel_case_valid = 2u) {
      return false;
    }
    for (size_t emel_case_valid = emel_branch_valid; emel_case_valid == 1u;
         emel_case_valid = 2u) {
      const float * a = static_cast<const float *>(request.src0.data);
      const float * b = static_cast<const float *>(request.src1.data);
      float * c = static_cast<float *>(request.dst.data);

      constexpr uint64_t row_block = 4;
      constexpr uint64_t col_vec = 8;
      constexpr uint64_t col_block = 64;
      constexpr uint64_t depth_block = 64;
      alignas(64) static thread_local float packed_b[depth_block * col_block];

      for (uint64_t jb = 0; jb < n; jb += col_block) {
        const uint64_t j_end = std::min<uint64_t>(n, jb + col_block);
        const uint64_t vec_cols = ((j_end - jb) / col_vec) * col_vec;
        const uint64_t j_vec_end = jb + vec_cols;

        for (uint64_t pb = 0; pb < k; pb += depth_block) {
          const uint64_t depth = std::min<uint64_t>(depth_block, k - pb);
          const bool first_depth_block = (pb == 0);

          {
            const size_t emel_branch_vec_cols = static_cast<size_t>(vec_cols != 0);
            for (size_t emel_case_vec_cols = emel_branch_vec_cols; emel_case_vec_cols == 1u;
                 emel_case_vec_cols = 2u) {
              for (uint64_t kk = 0; kk < depth; ++kk) {
                const float * b_src = b + (pb + kk) * n + jb;
                float * b_dst = packed_b + kk * vec_cols;
                std::memcpy(b_dst, b_src, static_cast<size_t>(vec_cols) * sizeof(float));
#if defined(__GNUC__) || defined(__clang__)
                {
                  const size_t emel_branch_prefetch =
                      static_cast<size_t>((kk & 15u) == 0 && kk + 16u < depth);
                  for (size_t emel_case_prefetch = emel_branch_prefetch;
                       emel_case_prefetch == 1u;
                       emel_case_prefetch = 2u) {
                    _mm_prefetch(
                        reinterpret_cast<const char *>(b + (pb + kk + 16u) * n + jb),
                        _MM_HINT_T0);
                  }
                  for (size_t emel_case_prefetch = emel_branch_prefetch;
                       emel_case_prefetch == 0u;
                       emel_case_prefetch = 2u) {

                  }
                }
#endif
              }

              for (uint64_t j = jb; j < j_vec_end; j += col_vec) {
                const uint64_t j_offset = j - jb;
                uint64_t i = 0;
                for (; i + row_block <= m; i += row_block) {
                  __m256 acc0 = _mm256_loadu_ps(c + (i + 0) * n + j);
                  __m256 acc1 = _mm256_loadu_ps(c + (i + 1) * n + j);
                  __m256 acc2 = _mm256_loadu_ps(c + (i + 2) * n + j);
                  __m256 acc3 = _mm256_loadu_ps(c + (i + 3) * n + j);
                  {
                    const size_t emel_branch_first_depth =
                        static_cast<size_t>(first_depth_block);
                    for (size_t emel_case_first_depth = emel_branch_first_depth;
                         emel_case_first_depth == 1u;
                         emel_case_first_depth = 2u) {
                      acc0 = _mm256_setzero_ps();
                      acc1 = _mm256_setzero_ps();
                      acc2 = _mm256_setzero_ps();
                      acc3 = _mm256_setzero_ps();
                    }
                    for (size_t emel_case_first_depth = emel_branch_first_depth;
                         emel_case_first_depth == 0u;
                         emel_case_first_depth = 2u) {

                    }
                  }

                  for (uint64_t kk = 0; kk < depth; ++kk) {
                    const __m256 bv = _mm256_loadu_ps(packed_b + kk * vec_cols + j_offset);
                    acc0 = _mm256_add_ps(
                        acc0, _mm256_mul_ps(_mm256_set1_ps(a[(i + 0) * k + pb + kk]), bv));
                    acc1 = _mm256_add_ps(
                        acc1, _mm256_mul_ps(_mm256_set1_ps(a[(i + 1) * k + pb + kk]), bv));
                    acc2 = _mm256_add_ps(
                        acc2, _mm256_mul_ps(_mm256_set1_ps(a[(i + 2) * k + pb + kk]), bv));
                    acc3 = _mm256_add_ps(
                        acc3, _mm256_mul_ps(_mm256_set1_ps(a[(i + 3) * k + pb + kk]), bv));
                  }

                  _mm256_storeu_ps(c + (i + 0) * n + j, acc0);
                  _mm256_storeu_ps(c + (i + 1) * n + j, acc1);
                  _mm256_storeu_ps(c + (i + 2) * n + j, acc2);
                  _mm256_storeu_ps(c + (i + 3) * n + j, acc3);
                }

                for (; i < m; ++i) {
                  __m256 acc = _mm256_loadu_ps(c + i * n + j);
                  {
                    const size_t emel_branch_first_depth =
                        static_cast<size_t>(first_depth_block);
                    for (size_t emel_case_first_depth = emel_branch_first_depth;
                         emel_case_first_depth == 1u;
                         emel_case_first_depth = 2u) {
                      acc = _mm256_setzero_ps();
                    }
                    for (size_t emel_case_first_depth = emel_branch_first_depth;
                         emel_case_first_depth == 0u;
                         emel_case_first_depth = 2u) {

                    }
                  }
                  for (uint64_t kk = 0; kk < depth; ++kk) {
                    const __m256 bv = _mm256_loadu_ps(packed_b + kk * vec_cols + j_offset);
                    acc = _mm256_add_ps(
                        acc, _mm256_mul_ps(_mm256_set1_ps(a[i * k + pb + kk]), bv));
                  }
                  _mm256_storeu_ps(c + i * n + j, acc);
                }
              }
            }
            for (size_t emel_case_vec_cols = emel_branch_vec_cols; emel_case_vec_cols == 0u;
                 emel_case_vec_cols = 2u) {

            }
          }

          for (uint64_t j = j_vec_end; j < j_end; ++j) {
            for (uint64_t i = 0; i < m; ++i) {
              float acc = c[i * n + j];
              {
                const size_t emel_branch_first_depth = static_cast<size_t>(first_depth_block);
                for (size_t emel_case_first_depth = emel_branch_first_depth;
                     emel_case_first_depth == 1u;
                     emel_case_first_depth = 2u) {
                  acc = 0.0f;
                }
                for (size_t emel_case_first_depth = emel_branch_first_depth;
                     emel_case_first_depth == 0u;
                     emel_case_first_depth = 2u) {

                }
              }
              for (uint64_t kk = 0; kk < depth; ++kk) {
                acc += a[i * k + pb + kk] * b[(pb + kk) * n + j];
              }
              c[i * n + j] = acc;
            }
          }
        }
      }

      return true;
    }
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

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_unary(const event::op_unary & request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count = ::emel::kernel::detail::tensor_element_count(request.dst);
  const float * src = static_cast<const float *>(request.src0.data);
  float * dst = static_cast<float *>(request.dst.data);
  const uint8_t subop_code = static_cast<uint8_t>(request.subop);
  const size_t is_abs =
      static_cast<size_t>(subop_code == static_cast<uint8_t>(event::unary_subop::abs));
  const size_t is_neg =
      static_cast<size_t>(subop_code == static_cast<uint8_t>(event::unary_subop::neg));
  const size_t is_relu =
      static_cast<size_t>(subop_code == static_cast<uint8_t>(event::unary_subop::relu));
  const size_t kernel_index = is_abs * 1u + is_neg * 2u + is_relu * 3u;
  using unary_kernel_t = void (*)(const float *, float *, uint64_t) noexcept;
  constexpr std::array<unary_kernel_t, 3> kernels = {
      execute_avx2_unary_abs,
      execute_avx2_unary_neg,
      execute_avx2_unary_relu,
  };

  bool executed = false;
  {
    const size_t emel_branch_has_kernel = static_cast<size_t>(kernel_index != 0);
    for (size_t emel_case_has_kernel = emel_branch_has_kernel; emel_case_has_kernel == 1u;
         emel_case_has_kernel = 2u) {
      kernels[kernel_index - 1u](src, dst, count);
      executed = true;
    }
    for (size_t emel_case_has_kernel = emel_branch_has_kernel; emel_case_has_kernel == 0u;
         emel_case_has_kernel = 2u) {

    }
  }
  return executed;
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
  const size_t simd_succeeded =
      static_cast<size_t>(can_use_avx2(request, ctx.avx2_available) && execute_simd(request));
  for (size_t emel_case_simd_succeeded = simd_succeeded; emel_case_simd_succeeded == 1u;
       emel_case_simd_succeeded = 2u) {
    return true;
  }
  for (size_t emel_case_simd_succeeded = simd_succeeded; emel_case_simd_succeeded == 0u;
       emel_case_simd_succeeded = 2u) {
    return ::emel::kernel::detail::execute_scalar(request);
  }
  return false;
#else
  (void) ctx;
  return ::emel::kernel::detail::execute_scalar(request);
#endif
}

}  // namespace emel::kernel::x86_64::detail
namespace emel::kernel::x86_64::action {

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

struct exec_dispatch {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_request & ev,
                  context & ctx) const noexcept {
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type>
struct exec_scalar_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    ::emel::kernel::detail::execute_scalar_unchecked(ev.request);
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type>
struct exec_simd_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    ::emel::kernel::x86_64::detail::execute_simd_unchecked(ev.request);
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
      detail::exec_scalar_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_RUN_TYPE)
#undef EMEL_KERNEL_DECLARE_RUN_TYPE

using exec_simd_op_dup_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_dup>;
using exec_simd_op_add_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_add>;
using exec_simd_op_sub_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_sub>;
using exec_simd_op_mul_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_mul>;
using exec_simd_op_div_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_div>;
using exec_simd_op_sqr_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_sqr>;
using exec_simd_op_sqrt_t = detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_sqrt>;
using exec_simd_op_mul_mat_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_mul_mat>;
using exec_simd_op_unary_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_unary>;

#define EMEL_KERNEL_DECLARE_REJECT_TYPE(op_name)                                      \
  using reject_invalid_##op_name##_t =                                                \
      detail::reject_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
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
inline constexpr exec_simd_op_mul_mat_t exec_simd_op_mul_mat{};
inline constexpr exec_simd_op_unary_t exec_simd_op_unary{};

#define EMEL_KERNEL_DEFINE_RUN_ACTION(op_name) \
  inline constexpr exec_##op_name##_t exec_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_RUN_ACTION)
#undef EMEL_KERNEL_DEFINE_RUN_ACTION

#define EMEL_KERNEL_DEFINE_REJECT_ACTION(op_name)            \
  inline constexpr reject_invalid_##op_name##_t reject_invalid_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_REJECT_ACTION)
#undef EMEL_KERNEL_DEFINE_REJECT_ACTION

inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::x86_64::action
