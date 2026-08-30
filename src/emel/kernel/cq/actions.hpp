#pragma once

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/kernel/cq/detail.hpp"
#include "emel/kernel/cq/events.hpp"

namespace emel::kernel::cq::action {

struct context {
  uint64_t scalar_calls = 0u;
  uint64_t avx2_calls = 0u;
  uint64_t prepare_calls = 0u;
  uint64_t prepared_calls = 0u;
  uint64_t quantize_calls = 0u;
};

inline void quantize_a8(const event::quantize_a8_request &request) noexcept {
  float absmax = 0.0f;
  for (const float value : request.input) {
    const float magnitude = std::abs(value);
    absmax = magnitude > absmax ? magnitude : absmax;
  }
  request.scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
  for (size_t i = 0u; i < request.input.size(); ++i) {
    const float scaled = request.input[i] / request.scale;
    const float rounded = std::nearbyint(scaled);
    const float clamped =
        rounded < -128.0f ? -128.0f : (rounded > 127.0f ? 127.0f : rounded);
    const int8_t quantized = static_cast<int8_t>(clamped);
    request.quantized[i] = quantized;
    request.dequantized[i] = static_cast<float>(quantized) * request.scale;
  }
}

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
inline void
execute_scalar_gemv_rows(const event::gemv_rows_request &request) noexcept {
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

inline void prepare_q4(const event::prepare_q4_request &request) noexcept {
  const auto &view = request.weights;
  const uint32_t out = view.shape[0];
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  const uint32_t in_pad = (in + group - 1u) / group * group;
  const size_t index_count = static_cast<size_t>(out) * in_pad;
  const size_t blocked_rows = out / 8u * 8u;
  const size_t blocked_count = blocked_rows * in_pad;
  const size_t norm_count = index_count / group;
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  for (size_t i = 0u; i < index_count; ++i)
    request.indices[i] =
        static_cast<uint8_t>(detail::unpack_index<4u>(base, i));
  for (size_t row = 0u; row < blocked_rows; row += 8u)
    for (size_t i = 0u; i < in_pad; ++i)
      for (size_t lane = 0u; lane < 8u; ++lane)
        request.indices_by_input8[row * in_pad + i * 8u + lane] =
            request.indices[(row + lane) * in_pad + i];
  const uint8_t *norms = base + index_count / 2u;
  for (size_t i = 0u; i < norm_count; ++i)
    request.norms[i] = detail::fp16_to_fp32(detail::load_u16(norms + i * 2u));
  request.prepared = event::prepared_q4_view{
      .source = static_cast<const uint8_t *>(view.data),
      .out = out,
      .in = in,
      .group = group,
      .in_pad = in_pad,
      .indices = request.indices.first(index_count),
      .indices_by_input8 = request.indices_by_input8.first(blocked_count),
      .norms = request.norms.first(norm_count)};
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot(const event::prepared_q4_view &view,
                          const std::span<const float> codebook_span,
                          const std::span<const float> activation_fwht,
                          const uint32_t row_begin, const uint32_t row_count,
                          const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const float *codebook = detail::codebook_for<4u>(codebook_span);
  const __m256 codebook_low = _mm256_loadu_ps(codebook);
  const __m256 codebook_high = _mm256_loadu_ps(codebook + 8u);
  const uint32_t groups_per_row = view.in_pad / view.group;
  for (uint32_t row = 0u; row < row_count; ++row) {
    const uint32_t source_row = row_begin + row;
    const uint8_t *indices =
        view.indices.data() + static_cast<size_t>(source_row) * view.in_pad;
    const float *norms =
        view.norms.data() + static_cast<size_t>(source_row) * groups_per_row;
    __m256 accum = _mm256_setzero_ps();
    for (uint32_t begin = 0u, group_index = 0u; begin < view.in_pad;
         begin += view.group, ++group_index) {
      const __m256 norm_v = _mm256_set1_ps(norms[group_index]);
      for (uint32_t i = 0u; i < view.group; i += 8u) {
        const __m128i index_bytes = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(indices + begin + i));
        const __m256i index_v = _mm256_cvtepu8_epi32(index_bytes);
        const __m256i table_index =
            _mm256_and_si256(index_v, _mm256_set1_epi32(7));
        const __m256 low_values =
            _mm256_permutevar8x32_ps(codebook_low, table_index);
        const __m256 high_values =
            _mm256_permutevar8x32_ps(codebook_high, table_index);
        const __m256 values = _mm256_blendv_ps(
            low_values, high_values,
            _mm256_castsi256_ps(_mm256_slli_epi32(index_v, 28)));
        const __m256 activation_v =
            _mm256_loadu_ps(activation_fwht.data() + begin + i);
        accum =
            _mm256_fmadd_ps(_mm256_mul_ps(values, norm_v), activation_v, accum);
      }
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accum);
    output[row] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                  lanes[5] + lanes[6] + lanes[7];
  }
#else
  (void)view;
  (void)codebook_span;
  (void)activation_fwht;
  (void)row_begin;
  (void)row_count;
  (void)output;
#endif
}

template <uint32_t Rows>
EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_avx2_dot_row_block(
    const event::prepared_q4_view &view,
    const std::span<const float> codebook_span,
    const std::span<const float> activation_fwht, const uint32_t row_begin,
    const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  static_assert(Rows == 4u || Rows == 8u);
  const float *codebook = detail::codebook_for<4u>(codebook_span);
  const __m256 codebook_low = _mm256_loadu_ps(codebook);
  const __m256 codebook_high = _mm256_loadu_ps(codebook + 8u);
  const __m256i low_table_mask = _mm256_set1_epi32(7);
  const uint32_t groups_per_row = view.in_pad / view.group;
  const uint8_t *row_indices[Rows];
  const float *row_norms[Rows];
  __m256 accum[Rows];
  for (uint32_t row = 0u; row < Rows; ++row) {
    const uint32_t source_row = row_begin + row;
    row_indices[row] =
        view.indices.data() + static_cast<size_t>(source_row) * view.in_pad;
    row_norms[row] =
        view.norms.data() + static_cast<size_t>(source_row) * groups_per_row;
    accum[row] = _mm256_setzero_ps();
  }
  for (uint32_t begin = 0u, group_index = 0u; begin < view.in_pad;
       begin += view.group, ++group_index) {
    __m256 norm[Rows];
    for (uint32_t row = 0u; row < Rows; ++row)
      norm[row] = _mm256_set1_ps(row_norms[row][group_index]);
    for (uint32_t i = 0u; i < view.group; i += 8u) {
      const __m256 activation =
          _mm256_loadu_ps(activation_fwht.data() + begin + i);
      for (uint32_t row = 0u; row < Rows; ++row) {
        const __m128i index_bytes = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(row_indices[row] + begin + i));
        const __m256i index = _mm256_cvtepu8_epi32(index_bytes);
        const __m256i table_index =
            _mm256_and_si256(index, low_table_mask);
        const __m256 values = _mm256_blendv_ps(
            _mm256_permutevar8x32_ps(codebook_low, table_index),
            _mm256_permutevar8x32_ps(codebook_high, table_index),
            _mm256_castsi256_ps(_mm256_slli_epi32(index, 28)));
        accum[row] = _mm256_fmadd_ps(_mm256_mul_ps(values, norm[row]),
                                     activation, accum[row]);
      }
    }
  }
  for (uint32_t row = 0u; row < Rows; ++row) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accum[row]);
    output[row] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                  lanes[5] + lanes[6] + lanes[7];
  }
#else
  (void)view;
  (void)codebook_span;
  (void)activation_fwht;
  (void)row_begin;
  (void)output;
#endif
}

template <uint32_t Rows>
EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_avx2_dot_blocked(
    const event::prepared_q4_view &view,
    const std::span<const float> codebook_span,
    const std::span<const float> activation_fwht,
    const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  static_assert(Rows == 4u || Rows == 8u);
  const uint32_t blocked_rows = view.out / Rows * Rows;
  for (uint32_t row = 0u; row < blocked_rows; row += Rows)
    execute_prepared_avx2_dot_row_block<Rows>(
        view, codebook_span, activation_fwht, row, output.subspan(row, Rows));
  if (blocked_rows < view.out)
    execute_prepared_avx2_dot(view, codebook_span, activation_fwht,
                              blocked_rows, view.out - blocked_rows,
                              output.subspan(blocked_rows));
#else
  (void)view;
  (void)codebook_span;
  (void)activation_fwht;
  (void)output;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot_blocked4(const event::prepared_q4_view &view,
                                   const std::span<const float> codebook_span,
                                   const std::span<const float> activation_fwht,
                                   const std::span<float> output) noexcept {
  execute_prepared_avx2_dot_blocked<4u>(view, codebook_span, activation_fwht,
                                        output);
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot_blocked8(const event::prepared_q4_view &view,
                                   const std::span<const float> codebook_span,
                                   const std::span<const float> activation_fwht,
                                   const std::span<float> output) noexcept {
  execute_prepared_avx2_dot_blocked<8u>(view, codebook_span, activation_fwht,
                                        output);
}
template <bool Rows>
EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_gemv(const event::prepared_q4_view &view,
                           const std::span<const float> codebook_span,
                           const std::span<const float> activation,
                           const uint32_t row_begin, const uint32_t row_count,
                           const std::span<float> output,
                           const std::span<float> workspace) noexcept {
  detail::compute_fwht_groups(activation, view.in, view.group,
                              workspace.first(view.in_pad));
  if constexpr (Rows)
    execute_prepared_avx2_dot(view, codebook_span, workspace.first(view.in_pad),
                              row_begin, row_count, output);
  else
    execute_prepared_avx2_dot(view, codebook_span, workspace.first(view.in_pad),
                              row_begin, row_count, output);
}

inline void execute_prepared_dequant_rows(
    const event::prepared_dequant_rows_request &request) noexcept {
  const auto &view = request.weights;
  const float *codebook = detail::codebook_for<4u>(request.codebook);
  const uint32_t groups_per_row = view.in_pad / view.group;
  alignas(32) float values[detail::k_max_group];
  for (uint32_t row = 0u; row < request.row_count; ++row) {
    const uint32_t source_row = request.row_begin + row;
    const uint8_t *indices =
        view.indices.data() + static_cast<size_t>(source_row) * view.in_pad;
    const float *norms =
        view.norms.data() + static_cast<size_t>(source_row) * groups_per_row;
    float *row_out = request.output.data() + static_cast<size_t>(row) * view.in;
    for (uint32_t begin = 0u, group_index = 0u; begin < view.in_pad;
         begin += view.group, ++group_index) {
      for (uint32_t i = 0u; i < view.group; ++i)
        values[i] = codebook[indices[begin + i]] * norms[group_index];
      detail::fwht(values, view.group);
      const uint32_t keep = begin + view.group <= view.in
                                ? view.group
                                : (view.in > begin ? view.in - begin : 0u);
      for (uint32_t i = 0u; i < keep; ++i)
        row_out[begin + i] = values[i] * request.scale;
    }
  }
}

template <uint32_t Bits>
EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_avx2_gemv(const event::gemv_request &request) noexcept {
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
#if defined(__x86_64__) || defined(_M_X64)
  __m256 q4_codebook_low{};
  __m256 q4_codebook_high{};
  __m128i q4_duplicate_mask{};
  __m128i q4_high_nibble_mask{};
  __m128i q4_nibble_mask{};
  if constexpr (Bits == 4u) {
    q4_codebook_low = _mm256_loadu_ps(codebook);
    q4_codebook_high = _mm256_loadu_ps(codebook + 8u);
    q4_duplicate_mask =
        _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    q4_high_nibble_mask =
        _mm_setr_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0);
    q4_nibble_mask = _mm_set1_epi8(0x0f);
  }
#endif
  for (uint32_t row = 0u; row < out; ++row) {
    const uint8_t *row_packed = base + static_cast<size_t>(row) * packed_row;
    const uint8_t *row_norms = norms + static_cast<size_t>(row) * norm_row;
    __m256 accum = _mm256_setzero_ps();
    for (uint32_t begin = 0u, group_index = 0u; begin < in_pad;
         begin += group, ++group_index) {
      const float norm =
          detail::fp16_to_fp32(detail::load_u16(row_norms + group_index * 2u));
      const __m256 norm_v = _mm256_set1_ps(norm);
      const uint8_t *group_packed = row_packed + group_index * group_bytes;
      uint32_t i = 0u;
      for (; i + 8u <= group; i += 8u) {
        const __m256 activation =
            _mm256_loadu_ps(request.workspace.data() + begin + i);
        __m256 values;
        if constexpr (Bits == 4u) {
          uint32_t packed_word = 0u;
          std::memcpy(&packed_word, group_packed + (i >> 1u),
                      sizeof(packed_word));
          const __m128i packed_v =
              _mm_cvtsi32_si128(static_cast<int32_t>(packed_word));
          const __m128i duplicated =
              _mm_shuffle_epi8(packed_v, q4_duplicate_mask);
          const __m128i low_nibbles = _mm_and_si128(duplicated, q4_nibble_mask);
          const __m128i high_nibbles =
              _mm_and_si128(_mm_srli_epi16(duplicated, 4), q4_nibble_mask);
          const __m128i indices_u8 =
              _mm_blendv_epi8(low_nibbles, high_nibbles, q4_high_nibble_mask);
          const __m256i index_v = _mm256_cvtepu8_epi32(indices_u8);
          const __m256i table_index =
              _mm256_and_si256(index_v, _mm256_set1_epi32(7));
          const __m256 low_values =
              _mm256_permutevar8x32_ps(q4_codebook_low, table_index);
          const __m256 high_values =
              _mm256_permutevar8x32_ps(q4_codebook_high, table_index);
          const __m256 high_table_mask =
              _mm256_castsi256_ps(_mm256_slli_epi32(index_v, 28));
          values = _mm256_blendv_ps(low_values, high_values, high_table_mask);
        } else {
          alignas(32) int32_t indices[8];
          for (uint32_t lane = 0u; lane < 8u; ++lane)
            indices[lane] = static_cast<int32_t>(
                detail::unpack_index<Bits>(group_packed, i + lane));
          const __m256i index_v =
              _mm256_load_si256(reinterpret_cast<const __m256i *>(indices));
          values = _mm256_i32gather_ps(codebook, index_v, 4);
        }
        accum =
            _mm256_fmadd_ps(_mm256_mul_ps(values, norm_v), activation, accum);
      }
      for (; i < group; ++i) {
        const uint32_t index = detail::unpack_index<Bits>(group_packed, i);
        request.output[row] +=
            detail::code_value<Bits>(index, group, request.codebook) * norm *
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
struct effect_quantize_a8 {
  void operator()(const event::quantize_a8 &ev, context &ctx) const noexcept {
    quantize_a8(ev.request);
    ev.result.accepted = true;
    ++ctx.quantize_calls;
  }
};

struct effect_prepare_q4 {
  void operator()(const event::prepare_q4 &ev, context &ctx) const noexcept {
    prepare_q4(ev.request);
    ev.result.accepted = true;
    ++ctx.prepare_calls;
  }
};

struct effect_execute_prepared_avx2_q4 {
  void operator()(const event::execute_prepared_avx2_q4 &ev,
                  context &ctx) const noexcept {
    for (uint32_t row = 0u; row < ev.request.weights.out; ++row)
      ev.request.output[row] = 0.0f;
    execute_prepared_avx2_gemv<false>(
        ev.request.weights, ev.request.codebook, ev.request.activation, 0u,
        ev.request.weights.out, ev.request.output, ev.request.workspace);
    ev.result.accepted = true;
    ++ctx.prepared_calls;
  }
};

struct effect_execute_prepared_avx2_batch4_q4 {
  void operator()(const event::execute_prepared_avx2_batch4_q4 &ev,
                  context &ctx) const noexcept {
    const auto &request = ev.request;
    const auto &first = *request.targets[0].weights;
    detail::compute_fwht_groups(request.activation, first.in, first.group,
                                request.workspace.first(first.in_pad));
    for (const auto &target : request.targets)
      execute_prepared_avx2_dot(*target.weights, request.codebook,
                                request.workspace.first(first.in_pad), 0u,
                                target.weights->out, target.output);
    ev.result.accepted = true;
    ctx.prepared_calls += request.targets.size();
  }
};

struct effect_execute_prepared_avx2_rows_q4 {
  void operator()(const event::execute_prepared_avx2_rows_q4 &ev,
                  context &ctx) const noexcept {
    for (uint32_t row = 0u; row < ev.request.row_count; ++row)
      ev.request.output[row] = 0.0f;
    execute_prepared_avx2_gemv<true>(ev.request.weights, ev.request.codebook,
                                     ev.request.activation,
                                     ev.request.row_begin, ev.request.row_count,
                                     ev.request.output, ev.request.workspace);
    ev.result.accepted = true;
    ++ctx.prepared_calls;
  }
};

struct effect_execute_prepared_dequant_q4 {
  void operator()(const event::execute_prepared_dequant_q4 &ev,
                  context &ctx) const noexcept {
    execute_prepared_dequant_rows(ev.request);
    ev.result.accepted = true;
    ++ctx.prepared_calls;
  }
};

template <uint32_t Bits> struct effect_execute_scalar {
  void operator()(const event::execute_scalar<Bits> &ev,
                  context &ctx) const noexcept {
    execute_scalar_gemv<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.scalar_calls;
  }
};

template <uint32_t Bits> struct effect_execute_avx2 {
  void operator()(const event::execute_avx2<Bits> &ev,
                  context &ctx) const noexcept {
    for (uint32_t row = 0u; row < ev.request.weights.shape[0]; ++row)
      ev.request.output[row] = 0.0f;
    execute_avx2_gemv<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.avx2_calls;
  }
};

template <uint32_t Bits> struct effect_execute_scalar_rows {
  void operator()(const event::execute_scalar_rows<Bits> &ev,
                  context &ctx) const noexcept {
    execute_scalar_gemv_rows<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.scalar_calls;
  }
};

template <uint32_t Bits> struct effect_execute_scalar_dequant {
  void operator()(const event::execute_scalar_dequant<Bits> &ev,
                  context &ctx) const noexcept {
    execute_scalar_dequant_rows<Bits>(ev.request);
    ev.result.accepted = true;
    ++ctx.scalar_calls;
  }
};

struct effect_capture_prepared_diagnostics {
  void operator()(const event::capture_prepared_diagnostics &ev,
                  const context &ctx) const noexcept {
    ev.prepare_calls = ctx.prepare_calls;
    ev.prepared_calls = ctx.prepared_calls;
  }
};

struct effect_capture_a8_diagnostics {
  void operator()(const event::capture_a8_diagnostics &ev,
                  const context &ctx) const noexcept {
    ev.quantize_calls = ctx.quantize_calls;
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
