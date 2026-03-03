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

  const bool base_supported = neon_available &&
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
  const bool valid_dims = k != 0 && m != 0 && n != 0;
  const bool valid_layout =
      request.src1.ne[1] == k && request.dst.ne[0] == n && request.dst.ne[1] == m;
  const bool valid = valid_dims && valid_layout;
  const uint64_t valid_u64 = static_cast<uint64_t>(valid);
  const float * a = static_cast<const float *>(request.src0.data);
  const float * b = static_cast<const float *>(request.src1.data);
  float * c = static_cast<float *>(request.dst.data);

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

struct exec_dispatch {
  void operator()(const ::emel::kernel::aarch64::event::dispatch_request & ev,
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
