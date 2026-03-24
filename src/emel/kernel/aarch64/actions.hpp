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

template <class request_type>
inline bool can_use_neon_flash_attn_ext(const request_type & request,
                                        const bool neon_available) noexcept {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  (void) request;
  (void) neon_available;
  return false;
#else
  return neon_available && ::emel::kernel::detail::can_run_flash_attn_ext(request);
#endif
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

template <class request_type>
inline bool run_flash_attn_ext_neon(const request_type & request,
                                    const bool neon_available,
                                    ::emel::kernel::detail::flash_attn_workspace & workspace) noexcept {
  if (!can_use_neon_flash_attn_ext(request, neon_available)) {
    return false;
  }

  const uint64_t kv_tokens = request.src1.ne[1];
  const uint64_t attn_width = ::emel::kernel::detail::flash_attn_total_tokens(request);
  if (kv_tokens > workspace.score_buffer.size()) {
    return false;
  }
  if (attn_width < kv_tokens || attn_width > workspace.score_buffer.size()) {
    return false;
  }

  if (workspace.prepared_tokens == kv_tokens) {
    ++workspace.reuse_count;
  } else {
    workspace.prepared_tokens = kv_tokens;
  }

  const uint64_t head_dim = request.src0.ne[0];
  const uint64_t head_count = request.src0.ne[2];
  const uint64_t kv_head_count = request.src1.ne[2];
  const uint64_t n_rep = head_count / kv_head_count;
  const float scale = ::emel::kernel::detail::flash_attn_scale(request);

  for (uint64_t head = 0; head < head_count; ++head) {
    const uint64_t kv_head = head / n_rep;
    const float * q = tensor_row_ptr(request.src0, 0u, head);
    float * dst = tensor_row_ptr_mut(request.dst, 0u, head);

    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const float * k = tensor_row_ptr(request.src1, token, kv_head);
      workspace.score_buffer[token] =
          ::emel::kernel::detail::dot_product_ggml_f16_scores(q, k, head_dim) * scale;
    }

    float max_score = workspace.score_buffer[0];
    for (uint64_t token = 1; token < kv_tokens; ++token) {
      max_score = std::max(max_score, workspace.score_buffer[token]);
    }

    double score_sum = 0.0;
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const float prob = std::exp(workspace.score_buffer[token] - max_score);
      workspace.score_buffer[token] = prob;
      score_sum += static_cast<double>(prob);
    }

    const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      float acc = 0.0f;
      for (uint64_t token = 0; token < kv_tokens; ++token) {
        const float weight = workspace.score_buffer[token] * inv_score_sum;
        const float * v = tensor_row_ptr(request.src2, token, kv_head);
        acc += weight * v[dim];
      }
      dst[dim] = acc;
    }
  }

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
  // Match the shared/reference accumulation order exactly. The block-summed variant drifts
  // across longer decode prefixes even though individual row spot checks stay close.
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

  const uint8x16_t m3b = vdupq_n_u8(0x03u);
  const int32x4_t zero = vdupq_n_s32(0);
  const uint8x16_t m0 = vdupq_n_u8(1u);
  const uint8x16_t m1 = vshlq_n_u8(m0, 1);
  const uint8x16_t m2 = vshlq_n_u8(m0, 2);
  const uint8x16_t m3 = vshlq_n_u8(m0, 3);

  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const float d = rhs[block].d * ::emel::kernel::detail::quant::fp16_to_fp32(lhs[block].d);

    const uint8_t * q3 = lhs[block].qs.data();
    const uint8_t * qh = lhs[block].hmask.data();
    const int8_t * q8 = rhs[block].qs.data();

    uint32_t aux[3] = {};
    uint32_t utmp[4] = {};
    std::memcpy(aux, lhs[block].scales.data(), lhs[block].scales.size());
    utmp[3] = ((aux[1] >> 4u) & kmask2) | (((aux[2] >> 6u) & kmask1) << 4u);
    utmp[2] = ((aux[0] >> 4u) & kmask2) | (((aux[2] >> 4u) & kmask1) << 4u);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2u) & kmask1) << 4u);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0u) & kmask1) << 4u);

    int8_t * scale = reinterpret_cast<int8_t *>(utmp);
    for (int idx = 0; idx < 16; ++idx) {
      scale[idx] = static_cast<int8_t>(scale[idx] - 32);
    }

    uint8x16x4_t q3h = {};
    int8x16x4_t q3bytes = {};
    uint8x16x2_t qhbits = load_u8x16x2(qh);

    int32_t isum = 0;
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

      if (j == 0u) {
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
  return ::emel::kernel::detail::dot_q6_k_q8_k_row_scalar(lhs, rhs, block_count);
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
      if (::emel::kernel::aarch64::detail::run_flash_attn_ext_neon(ev.request,
                                                                   ctx.neon_available,
                                                                   ctx.flash_attn_workspace)) {
        ++ctx.optimized_flash_dispatch_count;
        detail::mark_done(ev, ctx);
      } else if (::emel::kernel::detail::run_flash_attn_ext_with_workspace(
                     ev.request, ctx.flash_attn_workspace)) {
        ++ctx.shared_flash_dispatch_count;
        detail::mark_done(ev, ctx);
      } else {
        detail::mark_error(
            ev, ctx, static_cast<int32_t>(emel::error::cast(error::invalid_request)));
      }
    } else {
      if constexpr (std::is_same_v<request_type, ::emel::kernel::event::op_mul_mat>) {
        const uint8_t src0_type = ::emel::kernel::detail::dtype_code(ev.request.src0.type);
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
