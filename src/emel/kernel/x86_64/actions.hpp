#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
#define EMEL_KERNEL_X86_AVX2_FMA_TARGET __attribute__((target("avx2,fma")))
#define EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET                                   \
  __attribute__((target("avx2,fma,f16c")))
#else
#define EMEL_KERNEL_X86_AVX2_TARGET
#define EMEL_KERNEL_X86_AVX2_FMA_TARGET
#define EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
#endif
#else
#define EMEL_KERNEL_X86_AVX2_TARGET
#define EMEL_KERNEL_X86_AVX2_FMA_TARGET
#define EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
#endif

namespace emel::kernel::x86_64::detail {

namespace event= ::emel::kernel::event;

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

inline constexpr bool avx2_fma_intrinsics_compiled =
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
    true;
#else
    false;
#endif
#else
    false;
#endif

inline constexpr bool avx2_fma_f16c_intrinsics_compiled =
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||            \
    defined(__GNUC__) || defined(__clang__)
    true;
#else
    false;
#endif
#else
    false;
#endif

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type &tensor) noexcept {
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

inline bool
unary_subop_supported_simd(const event::unary_subop subop) noexcept {
  const auto subop_code = static_cast<uint8_t>(subop);
  return subop_code == static_cast<uint8_t>(event::unary_subop::abs) ||
         subop_code == static_cast<uint8_t>(event::unary_subop::neg) ||
         subop_code == static_cast<uint8_t>(event::unary_subop::relu);
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void execute_avx2_unary_abs(const float *src, float *dst,
                                   const uint64_t count) noexcept {
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
  (void)src;
  (void)dst;
  (void)count;
#endif
#else
  (void)src;
  (void)dst;
  (void)count;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void execute_avx2_unary_neg(const float *src, float *dst,
                                   const uint64_t count) noexcept {
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
  (void)src;
  (void)dst;
  (void)count;
#endif
#else
  (void)src;
  (void)dst;
  (void)count;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void execute_avx2_unary_relu(const float *src, float *dst,
                                    const uint64_t count) noexcept {
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
  (void)src;
  (void)dst;
  (void)count;
#endif
#else
  (void)src;
  (void)dst;
  (void)count;
#endif
}

template <class request_type>
inline bool can_use_avx2(const request_type &request,
                         const bool avx2_available) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)avx2_available;
  return false;
#else
  if constexpr (!simd_supported_request_v<request_type>) {
    return false;
  }

  const bool base_supported =
      avx2_available && avx2_intrinsics_compiled &&
      ::emel::kernel::detail::can_run_backend_request(request) &&
      ::emel::kernel::detail::dtype_code(request.src0.type) ==
          ::emel::kernel::detail::dtype_f32 &&
      ::emel::kernel::detail::dtype_code(request.dst.type) ==
          ::emel::kernel::detail::dtype_f32;
  bool src1_supported = true;
  if constexpr (::emel::kernel::detail::requires_src1_v<request_type>) {
    src1_supported = ::emel::kernel::detail::dtype_code(request.src1.type) ==
                         ::emel::kernel::detail::dtype_f32 &&
                     is_dense_contiguous(request.src1);
  }

  bool unary_supported = true;
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    unary_supported = unary_subop_supported_simd(request.subop);
  }

  return base_supported && src1_supported && unary_supported &&
         is_dense_contiguous(request.src0) && is_dense_contiguous(request.dst);
#endif
}

template <class request_type>
inline bool can_use_avx2_fma_f16c_flash_attn_ext_f16kv_one_chunk(
    const request_type &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  return host_features.avx2_fma_f16c_available() &&
         avx2_fma_f16c_intrinsics_compiled &&
         ::emel::kernel::detail::can_run_flash_attn_ext(request);
#endif
}

template <class request_type>
inline bool can_run_avx2_fma_f16c_flash_attn_ext_f16kv_one_chunk_request(
    const request_type &request, const host_feature_contract &host_features,
    const ::emel::kernel::detail::flash_attn_workspace &workspace) noexcept {
  return can_use_avx2_fma_f16c_flash_attn_ext_f16kv_one_chunk(request,
                                                              host_features) &&
         ::emel::kernel::detail::can_run_flash_attn_ext_with_workspace(
             request, workspace);
}

inline bool can_use_avx2_fma_q2_k_q8_k_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  return host_features.avx2_available &&
         host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         ::emel::kernel::detail::can_run_backend_request(request) &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_q2_k &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         k != 0u &&
         (k % ::emel::kernel::detail::quant::QK_K) == 0u &&
         block_count <= ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS;
#endif
}

inline bool can_use_avx2_fma_q3_k_q8_k_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  return host_features.avx2_available &&
         host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         ::emel::kernel::detail::can_run_backend_request(request) &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_q3_k &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         k != 0u &&
         (k % ::emel::kernel::detail::quant::QK_K) == 0u &&
         block_count <= ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS;
#endif
}

inline bool can_use_avx2_fma_f32_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  return host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         request.src1.ne[0] != 1u &&
         can_use_avx2(request, host_features.avx2_available);
#endif
}

inline bool can_use_avx2_fma_f32_vector_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  return host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         request.src1.ne[0] == 1u &&
         can_use_avx2(request, host_features.avx2_available);
#endif
}

inline bool can_use_avx2_fma_q4_k_q8_k_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  return host_features.avx2_available &&
         host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         ::emel::kernel::detail::can_run_backend_request(request) &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_q4_k &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         k != 0u &&
         (k % ::emel::kernel::detail::quant::QK_K) == 0u &&
         block_count <= ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS;
#endif
}

inline bool can_use_avx2_fma_q6_k_q8_k_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  return host_features.avx2_available &&
         host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         ::emel::kernel::detail::can_run_backend_request(request) &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             ::emel::kernel::detail::dtype_q6_k &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         k != 0u &&
         (k % ::emel::kernel::detail::quant::QK_K) == 0u &&
         block_count <= ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS;
#endif
}

template <uint8_t src0_dtype_code, uint64_t quant_block_size>
inline bool can_use_avx2_fma_q8_0_rhs_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
#if !(defined(__x86_64__) || defined(_M_X64))
  (void)request;
  (void)host_features;
  return false;
#else
  const uint64_t k = request.src0.ne[0];
  const uint64_t block_count = k / quant_block_size;
  return host_features.avx2_available &&
         host_features.fma_available &&
         avx2_fma_intrinsics_compiled &&
         ::emel::kernel::detail::can_run_backend_request(request) &&
         ::emel::kernel::detail::dtype_code(request.src0.type) ==
             src0_dtype_code &&
         ::emel::kernel::detail::dtype_code(request.src1.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         ::emel::kernel::detail::dtype_code(request.dst.type) ==
             ::emel::kernel::detail::dtype_f32 &&
         k != 0u &&
         (k % quant_block_size) == 0u &&
         block_count <= ::emel::kernel::detail::quant::MAX_Q8_0_BLOCKS;
#endif
}

inline bool can_use_avx2_fma_q4_0_q8_0_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
  return can_use_avx2_fma_q8_0_rhs_mul_mat<
      ::emel::kernel::detail::dtype_q4_0,
      ::emel::kernel::detail::quant::QK4_0>(request, host_features);
}

inline bool can_use_avx2_fma_q4_1_q8_0_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
  return can_use_avx2_fma_q8_0_rhs_mul_mat<
      ::emel::kernel::detail::dtype_q4_1,
      ::emel::kernel::detail::quant::QK4_1>(request, host_features);
}

inline bool can_use_avx2_fma_q5_0_q8_0_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
  return can_use_avx2_fma_q8_0_rhs_mul_mat<
      ::emel::kernel::detail::dtype_q5_0,
      ::emel::kernel::detail::quant::QK5_0>(request, host_features);
}

inline bool can_use_avx2_fma_q8_0_q8_0_mul_mat(
    const event::op_mul_mat &request,
    const host_feature_contract &host_features) noexcept {
  return can_use_avx2_fma_q8_0_rhs_mul_mat<
      ::emel::kernel::detail::dtype_q8_0,
      ::emel::kernel::detail::quant::QK8_0>(request, host_features);
}

#if defined(__x86_64__) || defined(_M_X64)
EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline int32_t horizontal_sum_i32x8_avx2(const __m256i values) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const __m128i low = _mm256_castsi256_si128(values);
  const __m128i high = _mm256_extracti128_si256(values, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
#else
  (void)values;
  return 0;
#endif
#else
  (void)values;
  return 0;
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline int32_t dot_u2_s8_16_avx2(const uint8_t *q2,
                                 const int8_t *q8,
                                 const int shift) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const __m128i q2_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(q2));
  const __m128i q8_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(q8));
  const __m256i q2_u16 = _mm256_and_si256(
      _mm256_srli_epi16(_mm256_cvtepu8_epi16(q2_bytes), shift),
      _mm256_set1_epi16(0x03));
  const __m256i q8_i16 = _mm256_cvtepi8_epi16(q8_bytes);
  return horizontal_sum_i32x8_avx2(_mm256_madd_epi16(q2_u16, q8_i16));
#else
  (void)q2;
  (void)q8;
  (void)shift;
  return 0;
#endif
#else
  (void)q2;
  (void)q8;
  (void)shift;
  return 0;
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline int32_t dot_q3_s8_16_avx2(const uint8_t *q3,
                                 const uint8_t *hmask,
                                 const int8_t *q8,
                                 const int shift,
                                 const uint8_t mask) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const __m128i q3_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(q3));
  const __m128i hmask_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(hmask));
  const __m128i q8_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(q8));

  const __m256i q3_u16 = _mm256_and_si256(
      _mm256_srli_epi16(_mm256_cvtepu8_epi16(q3_bytes), shift),
      _mm256_set1_epi16(0x03));
  const __m256i hmask_u16 = _mm256_cvtepu8_epi16(hmask_bytes);
  const __m256i missing_high_bit = _mm256_cmpeq_epi16(
      _mm256_and_si256(hmask_u16, _mm256_set1_epi16(mask)),
      _mm256_setzero_si256());
  const __m256i q3_i16 =
      _mm256_sub_epi16(q3_u16,
                       _mm256_and_si256(missing_high_bit,
                                        _mm256_set1_epi16(4)));
  const __m256i q8_i16 = _mm256_cvtepi8_epi16(q8_bytes);
  return horizontal_sum_i32x8_avx2(_mm256_madd_epi16(q3_i16, q8_i16));
#else
  (void)q3;
  (void)hmask;
  (void)q8;
  (void)shift;
  (void)mask;
  return 0;
#endif
#else
  (void)q3;
  (void)hmask;
  (void)q8;
  (void)shift;
  (void)mask;
  return 0;
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline int32_t dot_u6_s8_16_avx2(const uint8_t *ql,
                                 const uint8_t *qh,
                                 const int8_t *q8,
                                 const int low_shift,
                                 const int high_shift) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const __m128i ql_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql));
  const __m128i qh_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(qh));
  const __m128i q8_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(q8));

  const __m256i low_nibble = _mm256_and_si256(
      _mm256_srli_epi16(_mm256_cvtepu8_epi16(ql_bytes), low_shift),
      _mm256_set1_epi16(0x0f));
  const __m256i high_bits = _mm256_slli_epi16(
      _mm256_and_si256(
          _mm256_srli_epi16(_mm256_cvtepu8_epi16(qh_bytes), high_shift),
          _mm256_set1_epi16(0x03)),
      4);
  const __m256i q6_u16 = _mm256_or_si256(low_nibble, high_bits);
  const __m256i q8_i16 = _mm256_cvtepi8_epi16(q8_bytes);
  return horizontal_sum_i32x8_avx2(_mm256_madd_epi16(q6_u16, q8_i16));
#else
  (void)ql;
  (void)qh;
  (void)q8;
  (void)low_shift;
  (void)high_shift;
  return 0;
#endif
#else
  (void)ql;
  (void)qh;
  (void)q8;
  (void)low_shift;
  (void)high_shift;
  return 0;
#endif
}
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline __m256i unpack_nibbles_32_avx2(const uint8_t *src) noexcept {
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(src));
  const __m256i bytes = _mm256_insertf128_si256(
      _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
  return _mm256_and_si256(bytes, _mm256_set1_epi8(0x0f));
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline __m256i expand_high_bits_32_avx2(const uint8_t *qh) noexcept {
  uint32_t bits = 0u;
  std::memcpy(&bits, qh, sizeof(bits));
  const __m256i lane_bytes = _mm256_shuffle_epi8(
      _mm256_set1_epi32(static_cast<int32_t>(bits)),
      _mm256_set_epi64x(0x0303030303030303ll, 0x0202020202020202ll,
                        0x0101010101010101ll, 0x0000000000000000ll));
  const __m256i bit_select =
      _mm256_set1_epi64x(static_cast<int64_t>(0x7fbfdfeff7fbfdfeull));
  const __m256i selected = _mm256_or_si256(lane_bytes, bit_select);
  return _mm256_cmpeq_epi8(selected, _mm256_set1_epi64x(-1ll));
}

// Precondition: y lanes must be > -128. _mm256_sign_epi8 cannot negate -128
// (two's-complement wrap), so a -128 y lane paired with a negative x lane
// would flip the product sign. Every caller passes activations produced by
// quantize_row_q8_0_strided, which clamps to [-127, 127]; x (weight lanes,
// which may hold -128 in q8_0 model data) is only ever abs'd, where the
// u8 reinterpretation of -128 as 128 is exact.
EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline __m256i dot_i8_pairs_i32x8_avx2(const __m256i x,
                                       const __m256i y) noexcept {
  const __m256i abs_x = _mm256_sign_epi8(x, x);
  const __m256i signed_y = _mm256_sign_epi8(y, x);
  const __m256i pair_products = _mm256_maddubs_epi16(abs_x, signed_y);
  return _mm256_madd_epi16(_mm256_set1_epi16(1), pair_products);
}
#endif
#endif

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q2_k_q8_k_block_avx2_fma(
    const ::emel::kernel::detail::quant::block_q2_k &lhs,
    const ::emel::kernel::detail::quant::block_q8_k &rhs) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint8_t *q2 = lhs.qs.data();
  const int8_t *q8 = rhs.qs.data();
  const uint8_t *scales = lhs.scales.data();

  const __m128i scales_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(scales));
  const __m256i mins_i16 = _mm256_srli_epi16(
      _mm256_cvtepu8_epi16(scales_bytes), 4);
  const __m256i bsums_i16 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(rhs.bsums.data()));
  const int32_t sum_mins =
      horizontal_sum_i32x8_avx2(_mm256_madd_epi16(mins_i16, bsums_i16));

  int32_t sum = 0;
  int scale_index = 0;
  for (uint64_t block = 0;
       block < (::emel::kernel::detail::quant::QK_K / 128u); ++block) {
    for (int shift = 0; shift < 8; shift += 2) {
      sum += static_cast<int32_t>(scales[scale_index++] & 0x0fu) *
             dot_u2_s8_16_avx2(q2, q8, shift);
      sum += static_cast<int32_t>(scales[scale_index++] & 0x0fu) *
             dot_u2_s8_16_avx2(q2 + 16, q8 + 16, shift);
      q8 += 32;
    }
    q2 += 32;
  }

  const float d_all =
      rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d);
  const float d_min =
      rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.dmin);
  const __m128 block_sum = _mm_fmadd_ss(
      _mm_set_ss(d_all), _mm_set_ss(static_cast<float>(sum)),
      _mm_set_ss(-d_min * static_cast<float>(sum_mins)));
  return _mm_cvtss_f32(block_sum);
#else
  return ::emel::kernel::detail::dot_q2_k_q8_k_block_scalar(lhs, rhs);
#endif
#else
  return ::emel::kernel::detail::dot_q2_k_q8_k_block_scalar(lhs, rhs);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q2_k_q8_k_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q2_k *lhs,
    const ::emel::kernel::detail::quant::block_q8_k *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q2_k_q8_k_block_avx2_fma(lhs[block], rhs[block]);
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q2_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q2_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q3_k_q8_k_block_avx2_fma(
    const ::emel::kernel::detail::quant::block_q3_k &lhs,
    const ::emel::kernel::detail::quant::block_q8_k &rhs) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  uint32_t scale_words[4] = {};
  std::memcpy(scale_words, lhs.scales.data(), lhs.scales.size());
  const uint32_t tmp = scale_words[2];
  scale_words[3] =
      ((scale_words[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
  scale_words[2] =
      ((scale_words[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
  scale_words[1] =
      (scale_words[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);
  scale_words[0] =
      (scale_words[0] & kmask2) | (((tmp >> 0u) & kmask1) << 4u);

  auto *scale = reinterpret_cast<int8_t *>(scale_words);
  for (uint64_t idx = 0; idx < 16u; ++idx) {
    scale[idx] = static_cast<int8_t>(scale[idx] - 32);
  }

  const uint8_t *q3 = lhs.qs.data();
  const uint8_t *hmask = lhs.hmask.data();
  const int8_t *q8 = rhs.qs.data();
  int32_t isum = 0;
  uint8_t mask = 1u;
  int scale_index = 0;
  for (uint64_t block = 0;
       block < (::emel::kernel::detail::quant::QK_K / 128u); ++block) {
    for (int shift = 0; shift < 8; shift += 2) {
      isum += static_cast<int32_t>(scale[scale_index++]) *
              dot_q3_s8_16_avx2(q3, hmask, q8, shift, mask);
      isum += static_cast<int32_t>(scale[scale_index++]) *
              dot_q3_s8_16_avx2(q3 + 16, hmask + 16, q8 + 16, shift, mask);
      q8 += 32;
      mask = static_cast<uint8_t>(mask << 1u);
    }
    q3 += 32;
  }

  const float d =
      rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d);
  const __m128 block_sum =
      _mm_mul_ss(_mm_set_ss(d), _mm_set_ss(static_cast<float>(isum)));
  return _mm_cvtss_f32(block_sum);
#else
  return ::emel::kernel::detail::dot_q3_k_q8_k_block_scalar(lhs, rhs);
#endif
#else
  return ::emel::kernel::detail::dot_q3_k_q8_k_block_scalar(lhs, rhs);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q3_k_q8_k_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q3_k *lhs,
    const ::emel::kernel::detail::quant::block_q8_k *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q3_k_q8_k_block_avx2_fma(lhs[block], rhs[block]);
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q3_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q3_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q4_k_q8_k_block_avx2_fma(
    const ::emel::kernel::detail::quant::block_q4_k &lhs,
    const ::emel::kernel::detail::quant::block_q8_k &rhs) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  constexpr uint32_t kmask1 = 0x3f3f3f3fu;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;
  constexpr uint32_t kmask3 = 0x03030303u;
  alignas(32) static constexpr uint8_t SCALE_PAIR_SHUFFLE[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
  };

  uint32_t scale_words[4] = {};
  std::memcpy(scale_words, lhs.scales.data(), lhs.scales.size());
  scale_words[3] = ((scale_words[2] >> 4u) & kmask2) |
                   (((scale_words[1] >> 6u) & kmask3) << 4u);
  const uint32_t scale_high = scale_words[1] & kmask1;
  scale_words[1] = (scale_words[2] & kmask2) |
                   (((scale_words[0] >> 6u) & kmask3) << 4u);
  scale_words[2] = scale_high;
  scale_words[0] &= kmask1;

  const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
      _mm_set_epi32(static_cast<int32_t>(scale_words[3]),
                    static_cast<int32_t>(scale_words[2]),
                    static_cast<int32_t>(scale_words[1]),
                    static_cast<int32_t>(scale_words[0])));

  const __m256i bsums_i16 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(rhs.bsums.data()));
  const __m128i bsum_pairs =
      _mm_hadd_epi16(_mm256_castsi256_si128(bsums_i16),
                     _mm256_extracti128_si256(bsums_i16, 1));
  const __m128i min_products = _mm_madd_epi16(
      _mm256_extracti128_si256(mins_and_scales, 1), bsum_pairs);
  __m128i min_sum = _mm_hadd_epi32(min_products, min_products);
  min_sum = _mm_hadd_epi32(min_sum, min_sum);
  const int32_t sum_mins = _mm_cvtsi128_si32(min_sum);

  const __m128i scales_i16 = _mm256_castsi256_si128(mins_and_scales);
  const __m256i scales_vec = _mm256_set_m128i(scales_i16, scales_i16);
  const __m256i low_nibble_mask = _mm256_set1_epi8(0x0f);
  const uint8_t *q4 = lhs.qs.data();
  const int8_t *q8 = rhs.qs.data();
  __m256i sum_i32 = _mm256_setzero_si256();

  for (uint64_t pair = 0;
       pair < (::emel::kernel::detail::quant::QK_K / 64u); ++pair) {
    const __m256i scale_low = _mm256_shuffle_epi8(
        scales_vec,
        _mm256_load_si256(
            reinterpret_cast<const __m256i *>(SCALE_PAIR_SHUFFLE) +
            (2u * pair)));
    const __m256i scale_high_vec = _mm256_shuffle_epi8(
        scales_vec,
        _mm256_load_si256(
            reinterpret_cast<const __m256i *>(SCALE_PAIR_SHUFFLE) +
            (2u * pair + 1u)));

    const __m256i q4_bits =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
    q4 += 32;
    const __m256i q4_low = _mm256_and_si256(q4_bits, low_nibble_mask);
    const __m256i q4_high =
        _mm256_and_si256(_mm256_srli_epi16(q4_bits, 4), low_nibble_mask);

    const __m256i q8_low =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
    q8 += 32;
    const __m256i products_low = _mm256_madd_epi16(
        scale_low, _mm256_maddubs_epi16(q4_low, q8_low));

    const __m256i q8_high =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
    q8 += 32;
    const __m256i products_high = _mm256_madd_epi16(
        scale_high_vec, _mm256_maddubs_epi16(q4_high, q8_high));

    sum_i32 = _mm256_add_epi32(
        sum_i32, _mm256_add_epi32(products_low, products_high));
  }

  const int32_t sum = horizontal_sum_i32x8_avx2(sum_i32);
  const float d_all =
      rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d);
  const float d_min =
      rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.dmin);
  const __m128 block_sum = _mm_fmadd_ss(
      _mm_set_ss(d_all), _mm_set_ss(static_cast<float>(sum)),
      _mm_set_ss(-d_min * static_cast<float>(sum_mins)));
  return _mm_cvtss_f32(block_sum);
#else
  return ::emel::kernel::detail::dot_q4_k_q8_k_block_scalar(lhs, rhs);
#endif
#else
  return ::emel::kernel::detail::dot_q4_k_q8_k_block_scalar(lhs, rhs);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q4_k_q8_k_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q4_k *lhs,
    const ::emel::kernel::detail::quant::block_q8_k *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q4_k_q8_k_block_avx2_fma(lhs[block], rhs[block]);
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q4_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q4_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q4_0_q8_0_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q4_0 *lhs,
    const ::emel::kernel::detail::quant::block_q8_0 *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const __m256i nibbles = unpack_nibbles_32_avx2(lhs[block].qs.data());
    const __m256i x = _mm256_sub_epi8(nibbles, _mm256_set1_epi8(8));
    const __m256i y = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(rhs[block].qs.data()));
    const int32_t sumi =
        horizontal_sum_i32x8_avx2(dot_i8_pairs_i32x8_avx2(x, y));
    sum += static_cast<float>(sumi) *
           (::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d) *
            ::emel::kernel::detail::quant::fp16_to_fp32(rhs[block].d));
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q4_0_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q4_0_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q4_1_q8_0_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q4_1 *lhs,
    const ::emel::kernel::detail::quant::block_q8_0 *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const __m256i nibbles = unpack_nibbles_32_avx2(lhs[block].qs.data());
    const __m256i y = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(rhs[block].qs.data()));
    const int32_t sumi = horizontal_sum_i32x8_avx2(_mm256_madd_epi16(
        _mm256_set1_epi16(1), _mm256_maddubs_epi16(nibbles, y)));
    const int32_t rhs_sum = horizontal_sum_i32x8_avx2(_mm256_madd_epi16(
        _mm256_set1_epi16(1),
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), y)));
    const float rhs_d =
        ::emel::kernel::detail::quant::fp16_to_fp32(rhs[block].d);
    sum += rhs_d *
           (::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d) *
                static_cast<float>(sumi) +
            ::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].m) *
                static_cast<float>(rhs_sum));
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q4_1_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q4_1_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q5_0_q8_0_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q5_0 *lhs,
    const ::emel::kernel::detail::quant::block_q8_0 *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const __m256i nibbles = unpack_nibbles_32_avx2(lhs[block].qs.data());
    const __m256i high_set = expand_high_bits_32_avx2(lhs[block].qh.data());
    const __m256i x = _mm256_or_si256(
        nibbles, _mm256_andnot_si256(
                     high_set, _mm256_set1_epi8(static_cast<char>(0xf0))));
    const __m256i y = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(rhs[block].qs.data()));
    const int32_t sumi =
        horizontal_sum_i32x8_avx2(dot_i8_pairs_i32x8_avx2(x, y));
    sum += static_cast<float>(sumi) *
           (::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d) *
            ::emel::kernel::detail::quant::fp16_to_fp32(rhs[block].d));
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q5_0_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q5_0_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q8_0_q8_0_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q8_0 *lhs,
    const ::emel::kernel::detail::quant::block_q8_0 *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const __m256i x = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(lhs[block].qs.data()));
    const __m256i y = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(rhs[block].qs.data()));
    const int32_t sumi =
        horizontal_sum_i32x8_avx2(dot_i8_pairs_i32x8_avx2(x, y));
    sum += static_cast<float>(sumi) *
           (::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d) *
            ::emel::kernel::detail::quant::fp16_to_fp32(rhs[block].d));
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q8_0_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q8_0_q8_0_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q6_k_q8_k_block_avx2_fma(
    const ::emel::kernel::detail::quant::block_q6_k &lhs,
    const ::emel::kernel::detail::quant::block_q8_k &rhs) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const __m128i scales_bytes =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs.scales.data()));
  const __m256i scales_i16 = _mm256_cvtepi8_epi16(scales_bytes);
  const __m256i bsums_i16 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(rhs.bsums.data()));
  const int32_t sum_mins =
      horizontal_sum_i32x8_avx2(_mm256_madd_epi16(scales_i16, bsums_i16));

  const uint8_t *ql = lhs.ql.data();
  const uint8_t *qh = lhs.qh.data();
  const int8_t *q8 = rhs.qs.data();
  const int8_t *scale = lhs.scales.data();
  int32_t isum = 0;
  int scale_index = 0;
  for (uint64_t block = 0;
       block < (::emel::kernel::detail::quant::QK_K / 128u); ++block) {
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 0, qh + 0, q8 + 0, 0, 0);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 16, qh + 16, q8 + 16, 0, 0);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 32, qh + 0, q8 + 32, 0, 2);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 48, qh + 16, q8 + 48, 0, 2);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 0, qh + 0, q8 + 64, 4, 4);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 16, qh + 16, q8 + 80, 4, 4);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 32, qh + 0, q8 + 96, 4, 6);
    isum += static_cast<int32_t>(scale[scale_index++]) *
            dot_u6_s8_16_avx2(ql + 48, qh + 16, q8 + 112, 4, 6);
    ql += 64;
    qh += 32;
    q8 += 128;
  }

  const int32_t adjusted = isum - (32 * sum_mins);
  const float d =
      rhs.d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs.d);
  const __m128 block_sum =
      _mm_mul_ss(_mm_set_ss(d), _mm_set_ss(static_cast<float>(adjusted)));
  return _mm_cvtss_f32(block_sum);
#else
  return ::emel::kernel::detail::dot_q6_k_q8_k_block_scalar(lhs, rhs);
#endif
#else
  return ::emel::kernel::detail::dot_q6_k_q8_k_block_scalar(lhs, rhs);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline float dot_q6_k_q8_k_row_avx2_fma(
    const ::emel::kernel::detail::quant::block_q6_k *lhs,
    const ::emel::kernel::detail::quant::block_q8_k *rhs,
    const uint64_t block_count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q6_k_q8_k_block_avx2_fma(lhs[block], rhs[block]);
  }
  return sum;
#else
  return ::emel::kernel::detail::dot_q6_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
#else
  return ::emel::kernel::detail::dot_q6_k_q8_k_row_scalar(lhs, rhs,
                                                          block_count);
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline void execute_avx2_fma_mul_mat_q2_k_q8_k_unchecked(
    const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);
  const auto *a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  std::array<::emel::kernel::detail::quant::block_q8_k,
             ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
      q8_blocks = {};

  for (uint64_t j = 0; j < n; ++j) {
    for (uint64_t block = 0; block < block_count; ++block) {
      ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
          b + block * ::emel::kernel::detail::quant::QK_K * n + j, n,
          &q8_blocks[block], ::emel::kernel::detail::quant::QK_K);
    }
    for (uint64_t i = 0; i < m; ++i) {
      const auto *row = reinterpret_cast<
          const ::emel::kernel::detail::quant::block_q2_k *>(
          a + i * row_bytes);
      c[i * n + j] =
          dot_q2_k_q8_k_row_avx2_fma(row, q8_blocks.data(), block_count);
    }
  }
  return;
#endif
#endif
  (void)request;
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline void execute_avx2_fma_mul_mat_q3_k_q8_k_unchecked(
    const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);
  const auto *a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  std::array<::emel::kernel::detail::quant::block_q8_k,
             ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
      q8_blocks = {};

  for (uint64_t j = 0; j < n; ++j) {
    for (uint64_t block = 0; block < block_count; ++block) {
      ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
          b + block * ::emel::kernel::detail::quant::QK_K * n + j, n,
          &q8_blocks[block], ::emel::kernel::detail::quant::QK_K);
    }
    for (uint64_t i = 0; i < m; ++i) {
      const auto *row = reinterpret_cast<
          const ::emel::kernel::detail::quant::block_q3_k *>(
          a + i * row_bytes);
      c[i * n + j] =
          dot_q3_k_q8_k_row_avx2_fma(row, q8_blocks.data(), block_count);
    }
  }
  return;
#endif
#endif
  (void)request;
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline void execute_avx2_fma_mul_mat_q4_k_q8_k_unchecked(
    const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);
  const auto *a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  std::array<::emel::kernel::detail::quant::block_q8_k,
             ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
      q8_blocks = {};

  for (uint64_t j = 0; j < n; ++j) {
    for (uint64_t block = 0; block < block_count; ++block) {
      ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
          b + block * ::emel::kernel::detail::quant::QK_K * n + j, n,
          &q8_blocks[block], ::emel::kernel::detail::quant::QK_K);
    }
    for (uint64_t i = 0; i < m; ++i) {
      const auto *row = reinterpret_cast<
          const ::emel::kernel::detail::quant::block_q4_k *>(
          a + i * row_bytes);
      c[i * n + j] =
          dot_q4_k_q8_k_row_avx2_fma(row, q8_blocks.data(), block_count);
    }
  }
  return;
#endif
#endif
  (void)request;
}

template <class block_type, uint64_t quant_block_size,
          float (*row_dot)(const block_type *,
                           const ::emel::kernel::detail::quant::block_q8_0 *,
                           uint64_t)>
inline void execute_avx2_fma_mul_mat_q8_0_rhs_unchecked(
    const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const uint64_t block_count = k / quant_block_size;
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);
  const auto *a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  std::array<::emel::kernel::detail::quant::block_q8_0,
             ::emel::kernel::detail::quant::MAX_Q8_0_BLOCKS>
      q8_blocks = {};

  for (uint64_t j = 0; j < n; ++j) {
    ::emel::kernel::detail::quant::quantize_row_q8_0_strided(
        b + j, n, q8_blocks.data(), static_cast<int64_t>(k));
    for (uint64_t i = 0; i < m; ++i) {
      const auto *row =
          reinterpret_cast<const block_type *>(a + i * row_bytes);
      c[i * n + j] = row_dot(row, q8_blocks.data(), block_count);
    }
  }
  return;
#endif
#endif
  (void)request;
}

inline void execute_avx2_fma_mul_mat_q4_0_q8_0_unchecked(
    const event::op_mul_mat &request) noexcept {
  execute_avx2_fma_mul_mat_q8_0_rhs_unchecked<
      ::emel::kernel::detail::quant::block_q4_0,
      ::emel::kernel::detail::quant::QK4_0,
      &dot_q4_0_q8_0_row_avx2_fma>(request);
}

inline void execute_avx2_fma_mul_mat_q4_1_q8_0_unchecked(
    const event::op_mul_mat &request) noexcept {
  execute_avx2_fma_mul_mat_q8_0_rhs_unchecked<
      ::emel::kernel::detail::quant::block_q4_1,
      ::emel::kernel::detail::quant::QK4_1,
      &dot_q4_1_q8_0_row_avx2_fma>(request);
}

inline void execute_avx2_fma_mul_mat_q5_0_q8_0_unchecked(
    const event::op_mul_mat &request) noexcept {
  execute_avx2_fma_mul_mat_q8_0_rhs_unchecked<
      ::emel::kernel::detail::quant::block_q5_0,
      ::emel::kernel::detail::quant::QK5_0,
      &dot_q5_0_q8_0_row_avx2_fma>(request);
}

inline void execute_avx2_fma_mul_mat_q8_0_q8_0_unchecked(
    const event::op_mul_mat &request) noexcept {
  execute_avx2_fma_mul_mat_q8_0_rhs_unchecked<
      ::emel::kernel::detail::quant::block_q8_0,
      ::emel::kernel::detail::quant::QK8_0,
      &dot_q8_0_q8_0_row_avx2_fma>(request);
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline void execute_avx2_fma_mul_mat_q6_k_q8_k_unchecked(
    const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const uint64_t block_count = k / ::emel::kernel::detail::quant::QK_K;
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);
  const auto *a = static_cast<const uint8_t *>(request.src0.data);
  const size_t row_bytes = request.src0.nb[1];
  std::array<::emel::kernel::detail::quant::block_q8_k,
             ::emel::kernel::detail::quant::MAX_Q8_K_BLOCKS>
      q8_blocks = {};

  for (uint64_t j = 0; j < n; ++j) {
    for (uint64_t block = 0; block < block_count; ++block) {
      ::emel::kernel::detail::quant::quantize_row_q8_k_strided(
          b + block * ::emel::kernel::detail::quant::QK_K * n + j, n,
          &q8_blocks[block], ::emel::kernel::detail::quant::QK_K);
    }
    for (uint64_t i = 0; i < m; ++i) {
      const auto *row = reinterpret_cast<
          const ::emel::kernel::detail::quant::block_q6_k *>(
          a + i * row_bytes);
      c[i * n + j] =
          dot_q6_k_q8_k_row_avx2_fma(row, q8_blocks.data(), block_count);
    }
  }
  return;
#endif
#endif
  (void)request;
}

EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
inline void convert_f32_to_f16_buffer_avx2_f16c(const float *src, uint16_t *dst,
                                                const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__GNUC__) ||           \
    defined(__clang__)
  constexpr int round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    const __m256 fp32_0 = _mm256_loadu_ps(src + idx + 0u);
    const __m256 fp32_1 = _mm256_loadu_ps(src + idx + 8u);
    const __m128i fp16_0 = _mm256_cvtps_ph(fp32_0, round_mode);
    const __m128i fp16_1 = _mm256_cvtps_ph(fp32_1, round_mode);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + idx + 0u), fp16_0);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + idx + 8u), fp16_1);
  }
  for (; idx + 8u <= count; idx += 8u) {
    const __m256 fp32 = _mm256_loadu_ps(src + idx);
    const __m128i fp16 = _mm256_cvtps_ph(fp32, round_mode);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + idx), fp16);
  }
  for (; idx < count; ++idx) {
    dst[idx] = ::emel::kernel::detail::quant::fp32_to_fp16(src[idx]);
  }
  return;
#endif
#endif
  ::emel::kernel::detail::convert_f32_to_fp16_buffer_scalar(src, dst, count);
}

EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
inline void convert_f16_buffer_to_f32_avx2_f16c(const uint16_t *src, float *dst,
                                                const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__GNUC__) ||           \
    defined(__clang__)
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    const __m128i fp16_0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + idx + 0u));
    const __m128i fp16_1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + idx + 8u));
    _mm256_storeu_ps(dst + idx + 0u, _mm256_cvtph_ps(fp16_0));
    _mm256_storeu_ps(dst + idx + 8u, _mm256_cvtph_ps(fp16_1));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const __m128i fp16 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + idx));
    _mm256_storeu_ps(dst + idx, _mm256_cvtph_ps(fp16));
  }
  for (; idx < count; ++idx) {
    dst[idx] = ::emel::kernel::detail::quant::fp16_to_fp32(src[idx]);
  }
  return;
#endif
#endif
  ::emel::kernel::detail::convert_f16_buffer_to_f32_scalar(src, dst, count);
}

EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
inline float
dot_product_f16_f16_scores_avx2_fma(const uint16_t *lhs, const uint16_t *rhs,
                                    const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||            \
    defined(__GNUC__) || defined(__clang__)
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();
  uint64_t idx = 0u;
  for (; idx + 32u <= count; idx += 32u) {
    const __m256 lhs0 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs + idx + 0u)));
    const __m256 rhs0 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + idx + 0u)));
    const __m256 lhs1 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs + idx + 8u)));
    const __m256 rhs1 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + idx + 8u)));
    const __m256 lhs2 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs + idx + 16u)));
    const __m256 rhs2 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + idx + 16u)));
    const __m256 lhs3 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs + idx + 24u)));
    const __m256 rhs3 = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs + idx + 24u)));
    sum0 = _mm256_fmadd_ps(lhs0, rhs0, sum0);
    sum1 = _mm256_fmadd_ps(lhs1, rhs1, sum1);
    sum2 = _mm256_fmadd_ps(lhs2, rhs2, sum2);
    sum3 = _mm256_fmadd_ps(lhs3, rhs3, sum3);
  }

  double sumf = 0.0;
  const __m256 sum01 = _mm256_add_ps(sum0, sum1);
  const __m256 sum23 = _mm256_add_ps(sum2, sum3);
  const __m256 sum = _mm256_add_ps(sum01, sum23);
  alignas(32) float lanes[8] = {};
  _mm256_store_ps(lanes, sum);
  for (float lane : lanes) {
    sumf += static_cast<double>(lane);
  }
  for (; idx < count; ++idx) {
    sumf += static_cast<double>(
                ::emel::kernel::detail::quant::fp16_to_fp32(lhs[idx])) *
            static_cast<double>(
                ::emel::kernel::detail::quant::fp16_to_fp32(rhs[idx]));
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

EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
inline void scale_f32_avx2(float *data, const float scale,
                           const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const __m256 scale_v = _mm256_set1_ps(scale);
  uint64_t idx = 0u;
  for (; idx + 8u <= count; idx += 8u) {
    const __m256 data_v = _mm256_loadu_ps(data + idx);
    _mm256_storeu_ps(data + idx, _mm256_mul_ps(data_v, scale_v));
  }
  for (; idx < count; ++idx) {
    data[idx] *= scale;
  }
  return;
#endif
#endif
  ::emel::kernel::detail::scale_f32_scalar(data, scale, count);
}

EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
inline void scale_f16_buffer_avx2_f16c(uint16_t *data, const float scale,
                                       const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__F16C__)) || defined(__GNUC__) ||           \
    defined(__clang__)
  constexpr int round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  const float rounded_scale = ::emel::kernel::detail::round_fp16_scalar(scale);
  const __m256 scale_v = _mm256_set1_ps(rounded_scale);
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    const __m128i data0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + idx + 0u));
    const __m128i data1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + idx + 8u));
    const __m256 scaled0 = _mm256_mul_ps(_mm256_cvtph_ps(data0), scale_v);
    const __m256 scaled1 = _mm256_mul_ps(_mm256_cvtph_ps(data1), scale_v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(data + idx + 0u),
                     _mm256_cvtps_ph(scaled0, round_mode));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(data + idx + 8u),
                     _mm256_cvtps_ph(scaled1, round_mode));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const __m128i data_v =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(data + idx));
    const __m256 scaled = _mm256_mul_ps(_mm256_cvtph_ps(data_v), scale_v);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(data + idx),
                     _mm256_cvtps_ph(scaled, round_mode));
  }
  for (; idx < count; ++idx) {
    const float rounded_value =
        ::emel::kernel::detail::quant::fp16_to_fp32(data[idx]);
    data[idx] = ::emel::kernel::detail::quant::fp32_to_fp16(
        ::emel::kernel::detail::round_fp16_scalar(rounded_value *
                                                  rounded_scale));
  }
  return;
#endif
#endif
  ::emel::kernel::detail::scale_f16_buffer_scalar(data, scale, count);
}

EMEL_KERNEL_X86_AVX2_FMA_F16C_TARGET
inline void axpy_f16_buffer_avx2_fma_f16c(uint16_t *dst, const uint16_t *src,
                                          const float alpha,
                                          const uint64_t count) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||            \
    defined(__GNUC__) || defined(__clang__)
  constexpr int round_mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  const float rounded_alpha = ::emel::kernel::detail::round_fp16_scalar(alpha);
  const __m256 alpha_v = _mm256_set1_ps(rounded_alpha);
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    const __m128i dst0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + idx + 0u));
    const __m128i dst1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + idx + 8u));
    const __m128i src0 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + idx + 0u));
    const __m128i src1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + idx + 8u));
    const __m256 out0 =
        _mm256_fmadd_ps(_mm256_cvtph_ps(src0), alpha_v, _mm256_cvtph_ps(dst0));
    const __m256 out1 =
        _mm256_fmadd_ps(_mm256_cvtph_ps(src1), alpha_v, _mm256_cvtph_ps(dst1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + idx + 0u),
                     _mm256_cvtps_ph(out0, round_mode));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + idx + 8u),
                     _mm256_cvtps_ph(out1, round_mode));
  }
  for (; idx + 8u <= count; idx += 8u) {
    const __m128i dst_v =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(dst + idx));
    const __m128i src_v =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + idx));
    const __m256 out = _mm256_fmadd_ps(_mm256_cvtph_ps(src_v), alpha_v,
                                       _mm256_cvtph_ps(dst_v));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + idx),
                     _mm256_cvtps_ph(out, round_mode));
  }
  for (; idx < count; ++idx) {
    const float rounded_dst =
        ::emel::kernel::detail::quant::fp16_to_fp32(dst[idx]);
    const float rounded_src =
        ::emel::kernel::detail::quant::fp16_to_fp32(src[idx]);
    dst[idx] = ::emel::kernel::detail::quant::fp32_to_fp16(
        ::emel::kernel::detail::round_fp16_scalar(rounded_dst +
                                                  rounded_src * rounded_alpha));
  }
  return;
#endif
#endif
  ::emel::kernel::detail::axpy_f16_buffer_scalar(dst, src, alpha, count);
}

template <class request_type>
inline void prepare_flash_attn_ext_f16kv_one_chunk_workspace_avx2(
    const request_type &request,
    ::emel::kernel::detail::flash_attn_workspace &workspace) noexcept {
  const uint64_t kv_tokens =
      ::emel::kernel::detail::flash_attn_active_tokens(request);
  const bool reusing = workspace.prepared_tokens == kv_tokens;
  workspace.reuse_count += static_cast<uint64_t>(reusing);
  workspace.prepared_tokens = kv_tokens;
}

template <class request_type>
inline void run_flash_attn_ext_f16kv_one_chunk_avx2_fma_f16c_unchecked(
    const request_type &request,
    ::emel::kernel::detail::flash_attn_workspace &workspace) noexcept {
  const uint64_t kv_tokens =
      ::emel::kernel::detail::flash_attn_active_tokens(request);
  prepare_flash_attn_ext_f16kv_one_chunk_workspace_avx2(request, workspace);
  const uint64_t head_dim = request.src0.ne[0];
  const uint64_t head_count = request.src0.ne[2];
  const uint64_t kv_head_count = request.src1.ne[2];
  const float scale = ::emel::kernel::detail::flash_attn_scale(request);
  const uint64_t n_rep = head_count / kv_head_count;
  for (uint64_t head = 0u; head < head_count; ++head) {
    const uint64_t kv_head = head / n_rep;
    const float *q =
        ::emel::kernel::detail::tensor_row_ptr(request.src0, 0u, head);
    uint16_t *accum = workspace.accum_buffer_f16.data();
    float *dst =
        ::emel::kernel::detail::tensor_row_ptr_mut(request.dst, 0u, head);

    convert_f32_to_f16_buffer_avx2_f16c(q, workspace.q_buffer_f16.data(),
                                        head_dim);
    std::memset(accum, 0, sizeof(uint16_t) * head_dim);

    const auto *k_head_base = static_cast<const char *>(request.src1.data) +
                              kv_head * request.src1.nb[2];
    const auto *v_head_base = static_cast<const char *>(request.src2.data) +
                              kv_head * request.src2.nb[2];
    const uint64_t k_stride = request.src1.nb[1];
    const uint64_t v_stride = request.src2.nb[1];
    const char *k_ptr_bytes = k_head_base;
    const char *v_ptr_bytes = v_head_base;

    float score_sum = 0.0f;
    float max_score = -std::numeric_limits<float>::infinity();
    for (uint64_t token = 0u; token < kv_tokens; ++token) {
      const uint16_t *k = reinterpret_cast<const uint16_t *>(k_ptr_bytes);
      const float score = dot_product_f16_f16_scores_avx2_fma(
                              workspace.q_buffer_f16.data(), k, head_dim) *
                          scale;
      const float old_max = max_score;
      float max_scale = 1.0f;
      float value_scale = 1.0f;
      if (score > max_score) {
        max_score = score;
        max_scale = std::exp(old_max - max_score);
        scale_f16_buffer_avx2_f16c(accum, max_scale, head_dim);
      } else {
        value_scale = std::exp(score - max_score);
      }

      const uint16_t *v = reinterpret_cast<const uint16_t *>(v_ptr_bytes);
      axpy_f16_buffer_avx2_fma_f16c(accum, v, value_scale, head_dim);
      score_sum = score_sum * max_scale + value_scale;

      k_ptr_bytes += k_stride;
      v_ptr_bytes += v_stride;
    }

    convert_f16_buffer_to_f32_avx2_f16c(accum, dst, head_dim);
    if (score_sum == 0.0f) {
      std::fill_n(dst, head_dim, 0.0f);
    } else {
      scale_f32_avx2(dst, 1.0f / score_sum, head_dim);
    }
  }
}

template <class request_type>
inline void run_flash_attn_ext_avx2_fma_f16c_unchecked(
    const request_type &request,
    ::emel::kernel::detail::flash_attn_workspace &workspace) noexcept {
  run_flash_attn_ext_f16kv_one_chunk_avx2_fma_f16c_unchecked(request,
                                                             workspace);
}

template <class request_type>
inline bool run_flash_attn_ext_avx2_fma_f16c(
    const request_type &request, const host_feature_contract &host_features,
    ::emel::kernel::detail::flash_attn_workspace &workspace) noexcept {
  if (!can_run_avx2_fma_f16c_flash_attn_ext_f16kv_one_chunk_request(
          request, host_features, workspace)) {
    return false;
  }
  run_flash_attn_ext_avx2_fma_f16c_unchecked(request, workspace);
  return true;
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_dup(const event::op_dup &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *src = static_cast<const float *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_add(const event::op_add &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *lhs = static_cast<const float *>(request.src0.data);
  const float *rhs = static_cast<const float *>(request.src1.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_sub(const event::op_sub &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *lhs = static_cast<const float *>(request.src0.data);
  const float *rhs = static_cast<const float *>(request.src1.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_mul(const event::op_mul &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *lhs = static_cast<const float *>(request.src0.data);
  const float *rhs = static_cast<const float *>(request.src1.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_div(const event::op_div &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *lhs = static_cast<const float *>(request.src0.data);
  const float *rhs = static_cast<const float *>(request.src1.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_sqr(const event::op_sqr &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *src = static_cast<const float *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_sqrt(const event::op_sqrt &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *src = static_cast<const float *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);

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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline bool execute_avx2_mul_mat(const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool valid_dims = k != 0 && m != 0 && n != 0;
  const bool valid_layout = request.src1.ne[1] == k && request.dst.ne[0] == n &&
                            request.dst.ne[1] == m;
  const bool valid = valid_dims && valid_layout;
  const uint64_t valid_u64 = static_cast<uint64_t>(valid);
  const float *a = static_cast<const float *>(request.src0.data);
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);

  constexpr uint64_t row_block = 4;
  constexpr uint64_t col_vec = 8;
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
      const __m256 zero = _mm256_setzero_ps();
      const __m256 depth_reset_mask = _mm256_castsi256_ps(
          _mm256_set1_epi32(-static_cast<int32_t>(first_depth_block)));

      for (uint64_t kk = 0; kk < depth; ++kk) {
        const float *b_src = b + (pb + kk) * n + jb;
        float *b_dst = packed_b + kk * vec_cols;
        std::memcpy(b_dst, b_src,
                    static_cast<size_t>(vec_cols) * sizeof(float));
#if defined(__GNUC__) || defined(__clang__)
        const uint64_t prefetch_distance =
            16u * static_cast<uint64_t>((kk & 15u) == 0u && kk + 16u < depth);
        _mm_prefetch(reinterpret_cast<const char *>(
                         b + (pb + kk + prefetch_distance) * n + jb),
                     _MM_HINT_T0);
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
          acc0 = _mm256_blendv_ps(acc0, zero, depth_reset_mask);
          acc1 = _mm256_blendv_ps(acc1, zero, depth_reset_mask);
          acc2 = _mm256_blendv_ps(acc2, zero, depth_reset_mask);
          acc3 = _mm256_blendv_ps(acc3, zero, depth_reset_mask);

          for (uint64_t kk = 0; kk < depth; ++kk) {
            const __m256 bv =
                _mm256_loadu_ps(packed_b + kk * vec_cols + j_offset);
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_mul_ps(_mm256_set1_ps(a[(i + 0) * k + pb + kk]), bv));
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_mul_ps(_mm256_set1_ps(a[(i + 1) * k + pb + kk]), bv));
            acc2 = _mm256_add_ps(
                acc2,
                _mm256_mul_ps(_mm256_set1_ps(a[(i + 2) * k + pb + kk]), bv));
            acc3 = _mm256_add_ps(
                acc3,
                _mm256_mul_ps(_mm256_set1_ps(a[(i + 3) * k + pb + kk]), bv));
          }

          _mm256_storeu_ps(c + (i + 0) * n + j, acc0);
          _mm256_storeu_ps(c + (i + 1) * n + j, acc1);
          _mm256_storeu_ps(c + (i + 2) * n + j, acc2);
          _mm256_storeu_ps(c + (i + 3) * n + j, acc3);
        }

        for (; i < m; ++i) {
          __m256 acc = _mm256_loadu_ps(c + i * n + j);
          acc = _mm256_blendv_ps(acc, zero, depth_reset_mask);
          for (uint64_t kk = 0; kk < depth; ++kk) {
            const __m256 bv =
                _mm256_loadu_ps(packed_b + kk * vec_cols + j_offset);
            acc = _mm256_add_ps(
                acc, _mm256_mul_ps(_mm256_set1_ps(a[i * k + pb + kk]), bv));
          }
          _mm256_storeu_ps(c + i * n + j, acc);
        }
      }

      const uint32_t keep_existing_mask =
          static_cast<uint32_t>(-static_cast<int32_t>(!first_depth_block));
      for (uint64_t j = j_vec_end; j < j_end; ++j) {
        for (uint64_t i = 0; i < m; ++i) {
          uint32_t acc_bits = 0u;
          std::memcpy(&acc_bits, c + i * n + j, sizeof(acc_bits));
          acc_bits &= keep_existing_mask;
          float acc = 0.0f;
          std::memcpy(&acc, &acc_bits, sizeof(acc));
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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline bool
execute_avx2_fma_mul_mat(const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool valid_dims = k != 0 && m != 0 && n != 0;
  const bool valid_layout = request.src1.ne[1] == k && request.dst.ne[0] == n &&
                            request.dst.ne[1] == m;
  const bool valid = valid_dims && valid_layout;
  const uint64_t valid_u64 = static_cast<uint64_t>(valid);
  const float *a = static_cast<const float *>(request.src0.data);
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);

  constexpr uint64_t row_block = 4;
  constexpr uint64_t col_vec = 8;
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
      const __m256 zero = _mm256_setzero_ps();
      const __m256 depth_reset_mask = _mm256_castsi256_ps(
          _mm256_set1_epi32(-static_cast<int32_t>(first_depth_block)));

      for (uint64_t kk = 0; kk < depth; ++kk) {
        const float *b_src = b + (pb + kk) * n + jb;
        float *b_dst = packed_b + kk * vec_cols;
        std::memcpy(b_dst, b_src,
                    static_cast<size_t>(vec_cols) * sizeof(float));
#if defined(__GNUC__) || defined(__clang__)
        const uint64_t prefetch_distance =
            16u * static_cast<uint64_t>((kk & 15u) == 0u && kk + 16u < depth);
        _mm_prefetch(reinterpret_cast<const char *>(
                         b + (pb + kk + prefetch_distance) * n + jb),
                     _MM_HINT_T0);
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
          acc0 = _mm256_blendv_ps(acc0, zero, depth_reset_mask);
          acc1 = _mm256_blendv_ps(acc1, zero, depth_reset_mask);
          acc2 = _mm256_blendv_ps(acc2, zero, depth_reset_mask);
          acc3 = _mm256_blendv_ps(acc3, zero, depth_reset_mask);

          for (uint64_t kk = 0; kk < depth; ++kk) {
            const __m256 bv =
                _mm256_loadu_ps(packed_b + kk * vec_cols + j_offset);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 0) * k + pb + kk]),
                                   bv, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 1) * k + pb + kk]),
                                   bv, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 2) * k + pb + kk]),
                                   bv, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a[(i + 3) * k + pb + kk]),
                                   bv, acc3);
          }

          _mm256_storeu_ps(c + (i + 0) * n + j, acc0);
          _mm256_storeu_ps(c + (i + 1) * n + j, acc1);
          _mm256_storeu_ps(c + (i + 2) * n + j, acc2);
          _mm256_storeu_ps(c + (i + 3) * n + j, acc3);
        }

        for (; i < m; ++i) {
          __m256 acc = _mm256_loadu_ps(c + i * n + j);
          acc = _mm256_blendv_ps(acc, zero, depth_reset_mask);
          for (uint64_t kk = 0; kk < depth; ++kk) {
            const __m256 bv =
                _mm256_loadu_ps(packed_b + kk * vec_cols + j_offset);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(a[i * k + pb + kk]), bv, acc);
          }
          _mm256_storeu_ps(c + i * n + j, acc);
        }
      }

      const uint32_t keep_existing_mask =
          static_cast<uint32_t>(-static_cast<int32_t>(!first_depth_block));
      for (uint64_t j = j_vec_end; j < j_end; ++j) {
        for (uint64_t i = 0; i < m; ++i) {
          uint32_t acc_bits = 0u;
          std::memcpy(&acc_bits, c + i * n + j, sizeof(acc_bits));
          acc_bits &= keep_existing_mask;
          float acc = 0.0f;
          std::memcpy(&acc, &acc_bits, sizeof(acc));
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
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

EMEL_KERNEL_X86_AVX2_FMA_TARGET
inline void execute_avx2_fma_mul_mat_f32_vector_unchecked(
    const event::op_mul_mat &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__GNUC__) ||            \
    defined(__clang__)
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const float *a = static_cast<const float *>(request.src0.data);
  const float *b = static_cast<const float *>(request.src1.data);
  float *c = static_cast<float *>(request.dst.data);

  const uint64_t vec_depth = (k / 32u) * 32u;
  for (uint64_t i = 0; i < m; ++i) {
    const float *row = a + i * k;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (uint64_t kk = 0; kk < vec_depth; kk += 32u) {
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + kk),
                             _mm256_loadu_ps(b + kk), acc0);
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + kk + 8u),
                             _mm256_loadu_ps(b + kk + 8u), acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + kk + 16u),
                             _mm256_loadu_ps(b + kk + 16u), acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + kk + 24u),
                             _mm256_loadu_ps(b + kk + 24u), acc3);
    }
    uint64_t kk = vec_depth;
    for (; kk + 8u <= k; kk += 8u) {
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + kk),
                             _mm256_loadu_ps(b + kk), acc0);
    }
    const __m256 acc01 = _mm256_add_ps(acc0, acc1);
    const __m256 acc23 = _mm256_add_ps(acc2, acc3);
    const __m256 acc = _mm256_add_ps(acc01, acc23);
    const __m128 low = _mm256_castps256_ps128(acc);
    const __m128 high = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float total = _mm_cvtss_f32(sum);
    for (; kk < k; ++kk) {
      total += row[kk] * b[kk];
    }
    c[i] = total;
  }
  return;
#endif
#endif
  (void)request;
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void
execute_avx2_unary_abs_request(const event::op_unary &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *src = static_cast<const float *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);
  execute_avx2_unary_abs(src, dst, count);
#else
  (void)request;
#endif
#else
  (void)request;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void
execute_avx2_unary_neg_request(const event::op_unary &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *src = static_cast<const float *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);
  execute_avx2_unary_neg(src, dst, count);
#else
  (void)request;
#endif
#else
  (void)request;
#endif
}

EMEL_KERNEL_X86_AVX2_TARGET
inline void
execute_avx2_unary_relu_request(const event::op_unary &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
  const uint64_t count =
      ::emel::kernel::detail::tensor_element_count(request.dst);
  const float *src = static_cast<const float *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);
  execute_avx2_unary_relu(src, dst, count);
#else
  (void)request;
#endif
#else
  (void)request;
#endif
}

template <event::unary_subop subop>
inline void
execute_simd_unary_subop_unchecked(const event::op_unary &request) noexcept {
  if constexpr (subop == event::unary_subop::abs) {
    execute_avx2_unary_abs_request(request);
  }
  if constexpr (subop == event::unary_subop::neg) {
    execute_avx2_unary_neg_request(request);
  }
  if constexpr (subop == event::unary_subop::relu) {
    execute_avx2_unary_relu_request(request);
  }
}

template <class request_type>
inline void execute_simd_unchecked(const request_type &request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    (void)execute_avx2_dup(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_add>) {
    (void)execute_avx2_add(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sub>) {
    (void)execute_avx2_sub(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void)execute_avx2_mul(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_div>) {
    (void)execute_avx2_div(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    (void)execute_avx2_sqr(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    (void)execute_avx2_sqrt(request);
  }
  if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    (void)execute_avx2_mul_mat(request);
  }
}

template <class request_type>
inline bool execute_simd(const request_type &request) noexcept {
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
  return false;
}

template <class request_type, class context_type>
inline bool execute_request(const request_type &request,
                            const context_type &ctx) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const bool simd_succeeded =
      can_use_avx2(request, ctx.avx2_available) && execute_simd(request);
  return simd_succeeded || ::emel::kernel::detail::execute_scalar(request);
#else
  (void)ctx;
  return ::emel::kernel::detail::execute_scalar(request);
#endif
}

} // namespace emel::kernel::x86_64::detail
namespace emel::kernel::x86_64::action {

namespace detail {

template <class dispatch_event_type>
inline void mark_done(const dispatch_event_type &ev, context &ctx) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::done;
  ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::none));
}

template <class dispatch_event_type>
inline void mark_error(const dispatch_event_type &ev, context &ctx,
                       const int32_t err) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::failed;
  ev.ctx.err = err;
}

struct mark_done_op {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    mark_done(ev, ctx);
  }
};

struct exec_dispatch {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_request &ev,
                  context &ctx) const noexcept {
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type> struct exec_scalar_op {
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    using request_type = std::remove_cvref_t<decltype(ev.request)>;
    if constexpr (std::is_same_v<request_type,
                                 ::emel::kernel::event::op_flash_attn_ext>) {
      if (::emel::kernel::detail::run_flash_attn_ext_with_workspace(
              ev.request, ctx.flash_attn_workspace)) {
        ++ctx.shared_flash_dispatch_count;
        detail::mark_done(ev, ctx);
      } else {
        detail::mark_error(
            ev, ctx,
            static_cast<int32_t>(emel::error::cast(error::invalid_request)));
      }
    } else {
      if constexpr (std::is_same_v<request_type,
                                   ::emel::kernel::event::op_mul_mat>) {
        const uint8_t src0_type =
            ::emel::kernel::detail::dtype_code(ev.request.src0.type);
        ctx.shared_q2_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q2_k);
        ctx.shared_q3_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q3_k);
        ctx.shared_q4_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q4_k);
        ctx.shared_q6_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q6_k);
        ctx.shared_q4_0_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q4_0);
        ctx.shared_q4_1_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q4_1);
        ctx.shared_q5_0_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q5_0);
        ctx.shared_q8_0_dispatch_count += static_cast<uint64_t>(
            src0_type == ::emel::kernel::detail::dtype_q8_0);
      }
      ::emel::kernel::detail::execute_scalar_unchecked(ev.request);
      detail::mark_done(ev, ctx);
    }
  }
};

struct exec_simd_flash_attn_ext_f16kv_one_chunk {
  void operator()(
      const ::emel::kernel::x86_64::event::dispatch_op_flash_attn_ext &ev,
      context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        run_flash_attn_ext_f16kv_one_chunk_avx2_fma_f16c_unchecked(
            ev.request, ctx.flash_attn_workspace);
    ++ctx.optimized_flash_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q2_k_q8_k_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q2_k_q8_k_unchecked(ev.request);
    ++ctx.optimized_q2_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q3_k_q8_k_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q3_k_q8_k_unchecked(ev.request);
    ++ctx.optimized_q3_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_f32_fma_vector_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_f32_vector_unchecked(ev.request);
    ++ctx.optimized_f32_fma_vector_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_f32_fma_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::execute_avx2_fma_mul_mat(ev.request);
    ++ctx.optimized_f32_fma_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q4_k_q8_k_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q4_k_q8_k_unchecked(ev.request);
    ++ctx.optimized_q4_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q6_k_q8_k_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q6_k_q8_k_unchecked(ev.request);
    ++ctx.optimized_q6_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q4_0_q8_0_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q4_0_q8_0_unchecked(ev.request);
    ++ctx.optimized_q4_0_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q4_1_q8_0_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q4_1_q8_0_unchecked(ev.request);
    ++ctx.optimized_q4_1_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q5_0_q8_0_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q5_0_q8_0_unchecked(ev.request);
    ++ctx.optimized_q5_0_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

struct effect_exec_simd_q8_0_q8_0_op_mul_mat {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::
        execute_avx2_fma_mul_mat_q8_0_q8_0_unchecked(ev.request);
    ++ctx.optimized_q8_0_dispatch_count;
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type> struct exec_simd_op {
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::execute_simd_unchecked(ev.request);
    detail::mark_done(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop> struct exec_simd_unary_op {
  void operator()(const ::emel::kernel::x86_64::event::dispatch_op_unary &ev,
                  context &ctx) const noexcept {
    ::emel::kernel::x86_64::detail::execute_simd_unary_subop_unchecked<subop>(
        ev.request);
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type> struct reject_op {
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

} // namespace detail

using exec_dispatch_t = detail::exec_dispatch;

#define EMEL_KERNEL_DECLARE_RUN_TYPE(op_name)                                  \
  using exec_##op_name##_t = detail::exec_scalar_op<                           \
      ::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_RUN_TYPE)
#undef EMEL_KERNEL_DECLARE_RUN_TYPE

using exec_simd_op_dup_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_dup>;
using exec_simd_op_add_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_add>;
using exec_simd_op_sub_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_sub>;
using exec_simd_op_mul_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_mul>;
using exec_simd_op_div_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_div>;
using exec_simd_op_sqr_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_sqr>;
using exec_simd_op_sqrt_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_sqrt>;
using exec_simd_op_mul_mat_t =
    detail::exec_simd_op<::emel::kernel::x86_64::event::dispatch_op_mul_mat>;
using exec_simd_op_unary_abs_t =
    detail::exec_simd_unary_op<::emel::kernel::event::unary_subop::abs>;
using exec_simd_op_unary_neg_t =
    detail::exec_simd_unary_op<::emel::kernel::event::unary_subop::neg>;
using exec_simd_op_unary_relu_t =
    detail::exec_simd_unary_op<::emel::kernel::event::unary_subop::relu>;
using exec_simd_op_flash_attn_ext_f16kv_one_chunk_t =
    detail::exec_simd_flash_attn_ext_f16kv_one_chunk;
using effect_exec_simd_op_mul_mat_q2_k_q8_k_t =
    detail::effect_exec_simd_q2_k_q8_k_op_mul_mat;
using effect_exec_simd_op_mul_mat_q3_k_q8_k_t =
    detail::effect_exec_simd_q3_k_q8_k_op_mul_mat;
using effect_exec_simd_op_mul_mat_f32_fma_vector_t =
    detail::effect_exec_simd_f32_fma_vector_op_mul_mat;
using effect_exec_simd_op_mul_mat_f32_fma_t =
    detail::effect_exec_simd_f32_fma_op_mul_mat;
using effect_exec_simd_op_mul_mat_q4_k_q8_k_t =
    detail::effect_exec_simd_q4_k_q8_k_op_mul_mat;
using effect_exec_simd_op_mul_mat_q6_k_q8_k_t =
    detail::effect_exec_simd_q6_k_q8_k_op_mul_mat;
using effect_exec_simd_op_mul_mat_q4_0_q8_0_t =
    detail::effect_exec_simd_q4_0_q8_0_op_mul_mat;
using effect_exec_simd_op_mul_mat_q4_1_q8_0_t =
    detail::effect_exec_simd_q4_1_q8_0_op_mul_mat;
using effect_exec_simd_op_mul_mat_q5_0_q8_0_t =
    detail::effect_exec_simd_q5_0_q8_0_op_mul_mat;
using effect_exec_simd_op_mul_mat_q8_0_q8_0_t =
    detail::effect_exec_simd_q8_0_q8_0_op_mul_mat;
using exec_scalar_op_unary_abs_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::x86_64::event::dispatch_op_unary, context,
    detail::mark_done_op, ::emel::kernel::event::unary_subop::abs>;
using exec_scalar_op_unary_neg_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::x86_64::event::dispatch_op_unary, context,
    detail::mark_done_op, ::emel::kernel::event::unary_subop::neg>;
using exec_scalar_op_unary_relu_t =
    ::emel::kernel::detail::exec_scalar_unary_op<
        ::emel::kernel::x86_64::event::dispatch_op_unary, context,
        detail::mark_done_op, ::emel::kernel::event::unary_subop::relu>;
using exec_scalar_op_unary_exp_t = ::emel::kernel::detail::exec_scalar_unary_op<
    ::emel::kernel::x86_64::event::dispatch_op_unary, context,
    detail::mark_done_op, ::emel::kernel::event::unary_subop::exp>;

#define EMEL_KERNEL_DECLARE_REJECT_TYPE(op_name)                               \
  using reject_invalid_##op_name##_t =                                         \
      detail::reject_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_REJECT_TYPE)
#undef EMEL_KERNEL_DECLARE_REJECT_TYPE

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &ctx) const noexcept {
    if constexpr (requires { ev.ctx; }) {
      detail::mark_error(
          ev, ctx,
          static_cast<int32_t>(emel::error::cast(error::internal_error)));
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
inline constexpr exec_simd_op_unary_abs_t exec_simd_op_unary_abs{};
inline constexpr exec_simd_op_unary_neg_t exec_simd_op_unary_neg{};
inline constexpr exec_simd_op_unary_relu_t exec_simd_op_unary_relu{};
inline constexpr exec_simd_op_flash_attn_ext_f16kv_one_chunk_t
    exec_simd_op_flash_attn_ext_f16kv_one_chunk{};
inline constexpr effect_exec_simd_op_mul_mat_q2_k_q8_k_t
    effect_exec_simd_op_mul_mat_q2_k_q8_k{};
inline constexpr effect_exec_simd_op_mul_mat_q3_k_q8_k_t
    effect_exec_simd_op_mul_mat_q3_k_q8_k{};
inline constexpr effect_exec_simd_op_mul_mat_f32_fma_vector_t
    effect_exec_simd_op_mul_mat_f32_fma_vector{};
inline constexpr effect_exec_simd_op_mul_mat_f32_fma_t
    effect_exec_simd_op_mul_mat_f32_fma{};
inline constexpr effect_exec_simd_op_mul_mat_q4_k_q8_k_t
    effect_exec_simd_op_mul_mat_q4_k_q8_k{};
inline constexpr effect_exec_simd_op_mul_mat_q6_k_q8_k_t
    effect_exec_simd_op_mul_mat_q6_k_q8_k{};
inline constexpr effect_exec_simd_op_mul_mat_q4_0_q8_0_t
    effect_exec_simd_op_mul_mat_q4_0_q8_0{};
inline constexpr effect_exec_simd_op_mul_mat_q4_1_q8_0_t
    effect_exec_simd_op_mul_mat_q4_1_q8_0{};
inline constexpr effect_exec_simd_op_mul_mat_q5_0_q8_0_t
    effect_exec_simd_op_mul_mat_q5_0_q8_0{};
inline constexpr effect_exec_simd_op_mul_mat_q8_0_q8_0_t
    effect_exec_simd_op_mul_mat_q8_0_q8_0{};
inline constexpr exec_scalar_op_unary_abs_t exec_scalar_op_unary_abs{};
inline constexpr exec_scalar_op_unary_neg_t exec_scalar_op_unary_neg{};
inline constexpr exec_scalar_op_unary_relu_t exec_scalar_op_unary_relu{};
inline constexpr exec_scalar_op_unary_exp_t exec_scalar_op_unary_exp{};

#define EMEL_KERNEL_DEFINE_RUN_ACTION(op_name)                                 \
  inline constexpr exec_##op_name##_t exec_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_RUN_ACTION)
#undef EMEL_KERNEL_DEFINE_RUN_ACTION

#define EMEL_KERNEL_DEFINE_REJECT_ACTION(op_name)                              \
  inline constexpr reject_invalid_##op_name##_t reject_invalid_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_REJECT_ACTION)
#undef EMEL_KERNEL_DEFINE_REJECT_ACTION

inline constexpr on_unexpected on_unexpected{};

} // namespace emel::kernel::x86_64::action
