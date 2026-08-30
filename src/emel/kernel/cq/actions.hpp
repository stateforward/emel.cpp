#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/kernel/cq/detail.hpp"
#include "emel/kernel/cq/events.hpp"

namespace emel::kernel::cq::action {

struct context {
  uint64_t scalar_calls = 0u;
  uint64_t avx2_calls = 0u;
};

template <uint32_t Bits>
inline void execute_scalar_gemv(const event::gemv_request &request) noexcept {
  const auto &view = request.weights;
  const uint32_t out = view.shape[0];
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  const uint32_t in_pad = (in + group - 1u) / group * group;
  detail::compute_fwht_groups(request.activation, in, group,
                              request.workspace.first(in_pad));
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  const size_t packed_row = detail::packed_row_bytes<Bits>(in_pad);
  const size_t norm_row = static_cast<size_t>(in_pad / group) * 2u;
  const uint8_t *norms = base + static_cast<size_t>(out) * packed_row;
  for (uint32_t row = 0u; row < out; ++row)
    request.output[row] = detail::dequant_dot_row<Bits>(
        base + static_cast<size_t>(row) * packed_row,
        norms + static_cast<size_t>(row) * norm_row, in, group,
        request.codebook, request.workspace.first(in_pad));
}

template <uint32_t Bits>
inline void execute_scalar_gemv_rows(const event::gemv_rows_request &request) noexcept {
  const auto &view = request.weights;
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  const uint32_t in_pad = (in + group - 1u) / group * group;
  detail::compute_fwht_groups(request.activation, in, group,
                              request.workspace.first(in_pad));
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  const size_t packed_row = detail::packed_row_bytes<Bits>(in_pad);
  const size_t norm_row = static_cast<size_t>(in_pad / group) * 2u;
  const uint8_t *norms = base + static_cast<size_t>(view.shape[0]) * packed_row;
  for (uint32_t row = 0u; row < request.row_count; ++row) {
    const size_t src = static_cast<size_t>(request.row_begin) + row;
    request.output[row] = detail::dequant_dot_row<Bits>(
        base + src * packed_row, norms + src * norm_row, in, group,
        request.codebook, request.workspace.first(in_pad));
  }
}

template <uint32_t Bits>
inline void execute_scalar_dequant_rows(
    const event::dequant_rows_request &request) noexcept {
  const auto &view = request.weights;
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  const uint32_t in_pad = (in + group - 1u) / group * group;
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  const size_t packed_row = detail::packed_row_bytes<Bits>(in_pad);
  const size_t norm_row = static_cast<size_t>(in_pad / group) * 2u;
  const uint8_t *norms = base + static_cast<size_t>(view.shape[0]) * packed_row;
  for (uint32_t row = 0u; row < request.row_count; ++row) {
    const size_t src = static_cast<size_t>(request.row_begin) + row;
    detail::dequant_row_values<Bits>(
        base + src * packed_row, norms + src * norm_row, in, group,
        request.codebook, request.scale,
        request.output.data() + static_cast<size_t>(row) * in);
  }
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_CQ_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define EMEL_KERNEL_CQ_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_CQ_AVX2_TARGET
#endif

template <uint32_t Bits>
EMEL_KERNEL_CQ_AVX2_TARGET
inline void execute_avx2_gemv(const event::gemv_request &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const auto &view = request.weights;
  const uint32_t out = view.shape[0];
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  const uint32_t in_pad = (in + group - 1u) / group * group;
  detail::compute_fwht_groups(request.activation, in, group,
                              request.workspace.first(in_pad));
  const size_t packed_row = detail::packed_row_bytes<Bits>(in_pad);
  const size_t group_bytes = detail::packed_row_bytes<Bits>(group);
  const size_t norm_row = static_cast<size_t>(in_pad / group) * 2u;
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  const uint8_t *norms = base + static_cast<size_t>(out) * packed_row;
  const float *codebook = detail::codebook_for<Bits>(request.codebook);
  for (uint32_t row = 0u; row < out; ++row) {
    const uint8_t *row_packed = base + static_cast<size_t>(row) * packed_row;
    const uint8_t *row_norms = norms + static_cast<size_t>(row) * norm_row;
    __m256 accum = _mm256_setzero_ps();
    for (uint32_t begin = 0u, group_index = 0u; begin < in_pad;
         begin += group, ++group_index) {
      const float norm = detail::fp16_to_fp32(
          detail::load_u16(row_norms + group_index * 2u));
      const __m256 norm_v = _mm256_set1_ps(norm);
      const uint8_t *group_packed = row_packed + group_index * group_bytes;
      uint32_t i = 0u;
      for (; i + 8u <= group; i += 8u) {
        const __m256 activation = _mm256_loadu_ps(
            request.workspace.data() + begin + i);
        alignas(32) int32_t indices[8];
        for (uint32_t lane = 0u; lane < 8u; ++lane)
          indices[lane] = static_cast<int32_t>(detail::unpack_index<Bits>(
              group_packed, i + lane));
        const __m256i index_v = _mm256_load_si256(
            reinterpret_cast<const __m256i *>(indices));
        const __m256 values = _mm256_i32gather_ps(codebook, index_v, 4);
        accum = _mm256_fmadd_ps(_mm256_mul_ps(values, norm_v), activation,
                                accum);
      }
      for (; i < group; ++i) {
        const uint32_t index = detail::unpack_index<Bits>(group_packed, i);
        request.output[row] += detail::code_value<Bits>(
            index, group, request.codebook) * norm *
                               request.workspace[begin + i];
      }
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accum);
    request.output[row] += lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                           lanes[4] + lanes[5] + lanes[6] + lanes[7];
  }
#else
  execute_scalar_gemv<Bits>(request);
#endif
}
template <uint32_t Bits>
struct effect_execute_scalar {
  void operator()(const event::execute_scalar<Bits> &ev, context &ctx) const noexcept {
    execute_scalar_gemv<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.scalar_calls;
  }
};

template <uint32_t Bits>
struct effect_execute_avx2 {
  void operator()(const event::execute_avx2<Bits> &ev, context &ctx) const noexcept {
    for (uint32_t row = 0u; row < ev.request.weights.shape[0]; ++row)
      ev.request.output[row] = 0.0f;
    execute_avx2_gemv<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.avx2_calls;
  }
};

template <uint32_t Bits>
struct effect_execute_scalar_rows {
  void operator()(const event::execute_scalar_rows<Bits> &ev,
                  context &ctx) const noexcept {
    execute_scalar_gemv_rows<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.scalar_calls;
  }
};

template <uint32_t Bits>
struct effect_execute_scalar_dequant {
  void operator()(const event::execute_scalar_dequant<Bits> &ev,
                  context &ctx) const noexcept {
    execute_scalar_dequant_rows<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.scalar_calls;
  }
};

struct effect_capture_diagnostics {
  void operator()(const event::capture_diagnostics &ev,
                  const context &ctx) const noexcept {
    ev.scalar_calls = ctx.scalar_calls;
    ev.avx2_calls = ctx.avx2_calls;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::cq::action
