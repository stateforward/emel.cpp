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
  const size_t packed_row = static_cast<size_t>(in_pad) / 2u;
  const size_t blocked_rows = out / 8u * 8u;
  const size_t blocked_bytes = blocked_rows * packed_row;
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  for (size_t row = 0u; row < blocked_rows; row += 8u)
    for (size_t pair = 0u; pair < packed_row; ++pair)
      for (size_t lane = 0u; lane < 8u; ++lane)
        request.packed_by_pair8[row * packed_row + pair * 8u + lane] =
            base[(row + lane) * packed_row + pair];
  request.prepared = event::prepared_q4_view{
      .source = base,
      .out = out,
      .in = in,
      .group = group,
      .in_pad = in_pad,
      .packed_by_pair8 = request.packed_by_pair8.first(blocked_bytes)};
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
build_pair_lut(const std::span<const float> codebook_span, const float a0_value,
               const float a1_value, const std::span<float> pair_lut) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const float *codebook = detail::codebook_for<4u>(codebook_span);
  const __m256 a0 = _mm256_set1_ps(a0_value);
  const __m256 a1 = _mm256_set1_ps(a1_value);
  for (uint32_t hi = 0u; hi < 16u; ++hi) {
    const __m256 high = _mm256_mul_ps(a1, _mm256_set1_ps(codebook[hi]));
    _mm256_storeu_ps(pair_lut.data() + hi * 16u,
                     _mm256_fmadd_ps(a0, _mm256_loadu_ps(codebook), high));
    _mm256_storeu_ps(pair_lut.data() + hi * 16u + 8u,
                     _mm256_fmadd_ps(a0, _mm256_loadu_ps(codebook + 8u), high));
  }
#else
  (void)codebook_span;
  (void)a0_value;
  (void)a1_value;
  (void)pair_lut;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline __m256
load_group_norms8(const uint8_t *norms, const uint32_t groups_per_row,
                  const uint32_t group_index) noexcept {
  return _mm256_setr_ps(
      detail::fp16_to_fp32(detail::load_u16(norms + group_index * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (groups_per_row + group_index) * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (2u * groups_per_row + group_index) * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (3u * groups_per_row + group_index) * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (4u * groups_per_row + group_index) * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (5u * groups_per_row + group_index) * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (6u * groups_per_row + group_index) * 2u)),
      detail::fp16_to_fp32(
          detail::load_u16(norms + (7u * groups_per_row + group_index) * 2u)));
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
prepare_pair_values8(const event::prepared_q4_view &view,
                     const std::span<const float> codebook_span,
                     const std::span<const float> activation_fwht,
                     const std::span<float> pair_lut,
                     const std::span<float> pair_scratch) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const uint32_t pairs_per_row = view.in_pad / 2u;
  const uint32_t blocked_rows = view.out / 8u * 8u;
  for (uint32_t pair = 0u; pair < pairs_per_row; ++pair) {
    build_pair_lut(codebook_span, activation_fwht[pair * 2u],
                   activation_fwht[pair * 2u + 1u], pair_lut);
    for (uint32_t row = 0u; row < blocked_rows; row += 8u) {
      const uint8_t *packed = view.packed_by_pair8.data() +
                              static_cast<size_t>(row) * pairs_per_row;
      float *scratch =
          pair_scratch.data() + static_cast<size_t>(row) * pairs_per_row;
      const __m128i selector_bytes = _mm_loadl_epi64(
          reinterpret_cast<const __m128i *>(packed + pair * 8u));
      const __m256i selectors = _mm256_cvtepu8_epi32(selector_bytes);
      _mm256_storeu_ps(scratch + pair * 8u,
                       _mm256_i32gather_ps(pair_lut.data(), selectors, 4));
    }
    for (uint32_t row = blocked_rows; row < view.out; ++row) {
      const uint8_t *packed =
          view.source + static_cast<size_t>(row) * pairs_per_row;
      pair_scratch[static_cast<size_t>(row) * pairs_per_row + pair] =
          pair_lut[packed[pair]];
    }
  }
#else
  (void)view;
  (void)codebook_span;
  (void)activation_fwht;
  (void)pair_lut;
  (void)pair_scratch;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void prepare_pair_values_batch4(
    const event::prepared_gemv_batch4_request &request,
    const std::span<const float> activation_fwht) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const uint32_t pairs_per_row = request.targets[0].weights->in_pad / 2u;
  std::array<size_t, 4u> offsets{};
  for (size_t target_index = 1u; target_index < request.targets.size();
       ++target_index)
    offsets[target_index] =
        offsets[target_index - 1u] +
        static_cast<size_t>(request.targets[target_index - 1u].weights->out) *
            pairs_per_row;
  for (uint32_t pair = 0u; pair < pairs_per_row; ++pair) {
    build_pair_lut(request.codebook, activation_fwht[pair * 2u],
                   activation_fwht[pair * 2u + 1u], request.pair_lut);
    for (size_t target_index = 0u; target_index < request.targets.size();
         ++target_index) {
      const auto &view = *request.targets[target_index].weights;
      const uint32_t blocked_rows = view.out / 8u * 8u;
      float *target_scratch =
          request.pair_scratch.data() + offsets[target_index];
      for (uint32_t row = 0u; row < blocked_rows; row += 8u) {
        const uint8_t *packed = view.packed_by_pair8.data() +
                                static_cast<size_t>(row) * pairs_per_row;
        const __m128i selector_bytes = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(packed + pair * 8u));
        const __m256i selectors = _mm256_cvtepu8_epi32(selector_bytes);
        _mm256_storeu_ps(
            target_scratch + static_cast<size_t>(row) * pairs_per_row +
                pair * 8u,
            _mm256_i32gather_ps(request.pair_lut.data(), selectors, 4));
      }
      for (uint32_t row = blocked_rows; row < view.out; ++row) {
        const uint8_t *packed =
            view.source + static_cast<size_t>(row) * pairs_per_row;
        target_scratch[static_cast<size_t>(row) * pairs_per_row + pair] =
            request.pair_lut[packed[pair]];
      }
    }
  }
#else
  (void)request;
  (void)activation_fwht;
#endif
}
EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_pair_lut_dot_blocked8(
    const event::prepared_q4_view &view,
    const std::span<const float> pair_scratch,
    const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const uint32_t groups_per_row = view.in_pad / view.group;
  const uint32_t pairs_per_group = view.group / 2u;
  const uint32_t pairs_per_row = view.in_pad / 2u;
  const uint32_t blocked_rows = view.out / 8u * 8u;
  const size_t packed_row = pairs_per_row;
  const uint8_t *source_norms =
      view.source + static_cast<size_t>(view.out) * packed_row;
  for (uint32_t row = 0u; row < blocked_rows; row += 8u) {
    const float *scratch =
        pair_scratch.data() + static_cast<size_t>(row) * pairs_per_row;
    const uint8_t *norms =
        source_norms + static_cast<size_t>(row) * groups_per_row * 2u;
    __m256 accum = _mm256_setzero_ps();
    for (uint32_t group_index = 0u; group_index < groups_per_row;
         ++group_index) {
      __m256 group_sum = _mm256_setzero_ps();
      const uint32_t pair_begin = group_index * pairs_per_group;
      for (uint32_t pair = 0u; pair < pairs_per_group; ++pair)
        group_sum = _mm256_add_ps(
            group_sum, _mm256_loadu_ps(scratch + (pair_begin + pair) * 8u));
      accum = _mm256_fmadd_ps(
          group_sum, load_group_norms8(norms, groups_per_row, group_index),
          accum);
    }
    _mm256_storeu_ps(output.data() + row, accum);
  }
  for (uint32_t row = blocked_rows; row < view.out; ++row) {
    const uint8_t *norms =
        source_norms + static_cast<size_t>(row) * groups_per_row * 2u;
    const float *values =
        pair_scratch.data() + static_cast<size_t>(row) * pairs_per_row;
    float accum = 0.0f;
    for (uint32_t group_index = 0u; group_index < groups_per_row;
         ++group_index) {
      float group_sum = 0.0f;
      const uint32_t pair_begin = group_index * pairs_per_group;
      for (uint32_t pair = 0u; pair < pairs_per_group; ++pair)
        group_sum += values[pair_begin + pair];
      accum += group_sum *
               detail::fp16_to_fp32(detail::load_u16(norms + group_index * 2u));
    }
    output[row] = accum;
  }
#else
  (void)view;
  (void)pair_scratch;
  (void)output;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_pair_lut_dot_rows(
    const event::prepared_q4_view &view,
    const std::span<const float> pair_values, const uint32_t row_begin,
    const uint32_t row_count, const std::span<float> output) noexcept {
  const uint32_t groups_per_row = view.in_pad / view.group;
  const uint32_t pairs_per_group = view.group / 2u;
  const uint32_t pairs_per_row = view.in_pad / 2u;
  const size_t packed_row = pairs_per_row;
  const uint8_t *source_norms =
      view.source + static_cast<size_t>(view.out) * packed_row;
  for (uint32_t row = 0u; row < row_count; ++row) {
    const uint32_t source_row = row_begin + row;
    const uint8_t *norms =
        source_norms + static_cast<size_t>(source_row) * groups_per_row * 2u;
    const float *values =
        pair_values.data() + static_cast<size_t>(row) * pairs_per_row;
    float accum = 0.0f;
    for (uint32_t group_index = 0u; group_index < groups_per_row;
         ++group_index) {
      float group_sum = 0.0f;
      const uint32_t pair_begin = group_index * pairs_per_group;
      for (uint32_t pair = 0u; pair < pairs_per_group; ++pair)
        group_sum += values[pair_begin + pair];
      accum += group_sum *
               detail::fp16_to_fp32(detail::load_u16(norms + group_index * 2u));
    }
    output[row] = accum;
  }
}
inline void execute_prepared_dequant_rows(
    const event::prepared_dequant_rows_request &request) noexcept {
  const auto &view = request.weights;
  const float *codebook = detail::codebook_for<4u>(request.codebook);
  const uint32_t groups_per_row = view.in_pad / view.group;
  const size_t packed_row = static_cast<size_t>(view.in_pad) / 2u;
  const uint8_t *source_norms =
      view.source + static_cast<size_t>(view.out) * packed_row;
  alignas(32) float values[detail::k_max_group];
  for (uint32_t row = 0u; row < request.row_count; ++row) {
    const uint32_t source_row = request.row_begin + row;
    const uint8_t *packed =
        view.source + static_cast<size_t>(source_row) * packed_row;
    const uint8_t *norms =
        source_norms + static_cast<size_t>(source_row) * groups_per_row * 2u;
    float *row_out = request.output.data() + static_cast<size_t>(row) * view.in;
    for (uint32_t begin = 0u, group_index = 0u; begin < view.in_pad;
         begin += view.group, ++group_index) {
      const float norm = detail::fp16_to_fp32(
          detail::load_u16(norms + static_cast<size_t>(group_index) * 2u));
      for (uint32_t i = 0u; i < view.group; ++i) {
        const uint8_t byte = packed[(begin + i) / 2u];
        const uint32_t selector =
            (i & 1u) == 0u ? byte & 0x0fu : static_cast<uint32_t>(byte >> 4u);
        values[i] = codebook[selector] * norm;
      }
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
struct effect_prepare_q4 {
  void operator()(const event::prepare_q4 &ev, context &ctx) const noexcept {
    prepare_q4(ev.request);
    ev.result.accepted = true;
    ++ctx.prepare_calls;
  }
};

struct effect_execute_prepared_pair_lut_q4 {
  void operator()(const event::execute_prepared_pair_lut_q4 &ev,
                  context &ctx) const noexcept {
    const auto &view = ev.request.weights;
    detail::compute_fwht_groups(ev.request.activation, view.in, view.group,
                                ev.request.workspace.first(view.in_pad));
    prepare_pair_values8(view, ev.request.codebook,
                         ev.request.workspace.first(view.in_pad),
                         ev.request.pair_lut, ev.request.pair_scratch);
    execute_prepared_pair_lut_dot_blocked8(view, ev.request.pair_scratch,
                                           ev.request.output);
    ev.result.accepted = true;
    ++ctx.prepared_calls;
  }
};

struct effect_execute_prepared_pair_lut_batch4_q4 {
  void operator()(const event::execute_prepared_pair_lut_batch4_q4 &ev,
                  context &ctx) const noexcept {
    const auto &request = ev.request;
    const auto &first = *request.targets[0].weights;
    detail::compute_fwht_groups(request.activation, first.in, first.group,
                                request.workspace.first(first.in_pad));
    prepare_pair_values_batch4(request, request.workspace.first(first.in_pad));
    size_t scratch_offset = 0u;
    for (const auto &target : request.targets) {
      const size_t scratch_count =
          static_cast<size_t>(target.weights->out) * (first.in_pad / 2u);
      execute_prepared_pair_lut_dot_blocked8(
          *target.weights,
          request.pair_scratch.subspan(scratch_offset, scratch_count),
          target.output);
      scratch_offset += scratch_count;
    }
    ev.result.accepted = true;
    ctx.prepared_calls += request.targets.size();
  }
};

struct effect_execute_prepared_pair_lut_rows_q4 {
  void operator()(const event::execute_prepared_pair_lut_rows_q4 &ev,
                  context &ctx) const noexcept {
    const auto &view = ev.request.weights;
    detail::compute_fwht_groups(ev.request.activation, view.in, view.group,
                                ev.request.workspace.first(view.in_pad));
    const size_t pairs_per_row = static_cast<size_t>(view.in_pad) / 2u;
    for (size_t pair = 0u; pair < pairs_per_row; ++pair) {
      build_pair_lut(ev.request.codebook, ev.request.workspace[pair * 2u],
                     ev.request.workspace[pair * 2u + 1u], ev.request.pair_lut);
      for (uint32_t row = 0u; row < ev.request.row_count; ++row) {
        const uint8_t *selectors =
            view.source +
            static_cast<size_t>(ev.request.row_begin + row) * pairs_per_row;
        ev.request
            .pair_scratch[static_cast<size_t>(row) * pairs_per_row + pair] =
            ev.request.pair_lut[selectors[pair]];
      }
    }
    execute_prepared_pair_lut_dot_rows(view, ev.request.pair_scratch,
                                       ev.request.row_begin,
                                       ev.request.row_count, ev.request.output);
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
