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
  bool timing_enabled = false;
  event::timestamp_now_fn timing_now = nullptr;
  event::timing_breakdown timing = {};
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
    request.integer_values[i] = static_cast<float>(quantized);
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
    request.output[row] =
        detail::dequant_dot_row<Bits>(
            base + static_cast<size_t>(row) * packed_row,
            norms + static_cast<size_t>(row) * norm_row, in, group,
            request.codebook, request.workspace.first(in_pad)) *
        request.output_scale;
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
    request.output[row] =
        detail::dequant_dot_row<Bits>(
            base + src * packed_row, norms + src * norm_row, in, group,
            request.codebook, request.workspace.first(in_pad)) *
        request.output_scale;
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

inline void
prepare_codebook_q4(const event::prepare_codebook_q4_request &request) noexcept {
  request.prepared.values = request.codebook;
  const auto *codebook_bytes = reinterpret_cast<const uint8_t *>(
      detail::codebook_for<4u>(request.codebook));
  for (uint32_t index = 0u; index < 16u; ++index)
    for (uint32_t byte = 0u; byte < 4u; ++byte) {
      const uint8_t value = codebook_bytes[index * sizeof(float) + byte];
      request.prepared.byte_planes[byte][index] = value;
      request.prepared.byte_planes[byte][16u + index] = value;
    }
}

inline void prepare_q4(const event::prepare_q4_request &request) noexcept {
  const auto &view = request.weights;
  const uint32_t out = view.shape[0];
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  const uint32_t in_pad = (in + group - 1u) / group * group;
  const size_t index_count = static_cast<size_t>(out) * in_pad;
  const size_t blocked_rows = out / 32u * 32u;
  const size_t blocked_count = blocked_rows * in_pad;
  const size_t norm_count = index_count / group;
  const size_t groups_per_row = in_pad / group;
  const size_t blocked_norm_count = blocked_rows * groups_per_row;
  const uint8_t *base = static_cast<const uint8_t *>(view.data);
  for (size_t i = 0u; i < index_count; ++i)
    request.indices[i] =
        static_cast<uint8_t>(detail::unpack_index<4u>(base, i));
  for (size_t row = 0u; row < blocked_rows; row += 32u)
    for (size_t i = 0u; i < in_pad; ++i)
      for (size_t lane = 0u; lane < 32u; ++lane)
        request.indices_by_input32[row * in_pad + i * 32u + lane] =
            request.indices[(row + lane) * in_pad + i];
  const uint8_t *norms = base + index_count / 2u;
  for (size_t i = 0u; i < norm_count; ++i)
    request.norms[i] = detail::fp16_to_fp32(detail::load_u16(norms + i * 2u));
  constexpr std::array<size_t, 32u> lookup32_raw_rows{
      0u,  1u,  2u,  3u,  16u, 17u, 18u, 19u,
      4u,  5u,  6u,  7u,  20u, 21u, 22u, 23u,
      8u,  9u,  10u, 11u, 24u, 25u, 26u, 27u,
      12u, 13u, 14u, 15u, 28u, 29u, 30u, 31u};
  for (size_t row = 0u; row < blocked_rows; row += 32u)
    for (size_t group_index = 0u; group_index < groups_per_row;
         ++group_index)
      for (size_t lane = 0u; lane < lookup32_raw_rows.size(); ++lane)
        request.norms_by_group32[row * groups_per_row + group_index * 32u +
                                 lane] =
            request.norms[(row + lookup32_raw_rows[lane]) * groups_per_row +
                          group_index];
  request.prepared = event::prepared_q4_view{
      .source = static_cast<const uint8_t *>(view.data),
      .out = out,
      .in = in,
      .group = group,
      .in_pad = in_pad,
      .indices = request.indices.first(index_count),
      .indices_by_input32 = request.indices_by_input32.first(blocked_count),
      .norms = request.norms.first(norm_count),
      .norms_by_group32 =
          request.norms_by_group32.first(blocked_norm_count)};
}

#if defined(__x86_64__) || defined(_M_X64)
struct q4_lookup16_result {
  __m256 low;
  __m256 high;
};
struct q4_lookup32_result {
  __m256 values0;
  __m256 values1;
  __m256 values2;
  __m256 values3;
};

EMEL_KERNEL_CQ_AVX2_TARGET inline q4_lookup32_result
lookup_codebook32_raw(const __m256i index_bytes, const __m256i byte0,
                      const __m256i byte1, const __m256i byte2,
                      const __m256i byte3) noexcept {
  const __m256i bytes0 = _mm256_shuffle_epi8(byte0, index_bytes);
  const __m256i bytes1 = _mm256_shuffle_epi8(byte1, index_bytes);
  const __m256i bytes2 = _mm256_shuffle_epi8(byte2, index_bytes);
  const __m256i bytes3 = _mm256_shuffle_epi8(byte3, index_bytes);
  const __m256i low_words01 = _mm256_unpacklo_epi8(bytes0, bytes1);
  const __m256i low_words23 = _mm256_unpacklo_epi8(bytes2, bytes3);
  const __m256i high_words01 = _mm256_unpackhi_epi8(bytes0, bytes1);
  const __m256i high_words23 = _mm256_unpackhi_epi8(bytes2, bytes3);

  // AVX2 unpacks operate independently within each 128-bit lane. Therefore
  // the four natural results map to rows as follows before any cross-lane
  // permutation:
  //   low_values03:  0..3, 16..19
  //   low_values47:  4..7, 20..23
  //   high_values03: 8..11, 24..27
  //   high_values47: 12..15, 28..31
  // Block32 keeps this order through accumulation and uses identically ordered
  // prepared norms, so the hot per-input lookup performs no cross-lane work.
  return q4_lookup32_result{
      .values0 = _mm256_castsi256_ps(
          _mm256_unpacklo_epi16(low_words01, low_words23)),
      .values1 = _mm256_castsi256_ps(
          _mm256_unpackhi_epi16(low_words01, low_words23)),
      .values2 = _mm256_castsi256_ps(
          _mm256_unpacklo_epi16(high_words01, high_words23)),
      .values3 = _mm256_castsi256_ps(
          _mm256_unpackhi_epi16(high_words01, high_words23))};
}

EMEL_KERNEL_CQ_AVX2_TARGET inline q4_lookup32_result
lookup_codebook32_pshufb(const __m256i index_bytes, const __m256i byte0,
                         const __m256i byte1, const __m256i byte2,
                         const __m256i byte3) noexcept {
  const q4_lookup32_result raw = lookup_codebook32_raw(
      index_bytes, byte0, byte1, byte2, byte3);
  return q4_lookup32_result{
      .values0 = _mm256_permute2f128_ps(raw.values0, raw.values1, 0x20),
      .values1 = _mm256_permute2f128_ps(raw.values2, raw.values3, 0x20),
      .values2 = _mm256_permute2f128_ps(raw.values0, raw.values1, 0x31),
      .values3 = _mm256_permute2f128_ps(raw.values2, raw.values3, 0x31)};
}


EMEL_KERNEL_CQ_AVX2_TARGET inline q4_lookup16_result
lookup_codebook16_pshufb(const __m256i index_bytes, const __m256i byte0,
                         const __m256i byte1, const __m256i byte2,
                         const __m256i byte3) noexcept {
  // vpshufb indexes each 128-bit lane independently. The byte-plane tables
  // are duplicated across lanes, while the selector vector carries values
  // 0..7 in its low lane and 8..15 in its high lane.
  const __m256i values0 = _mm256_shuffle_epi8(byte0, index_bytes);
  const __m256i values1 = _mm256_shuffle_epi8(byte1, index_bytes);
  const __m256i values2 = _mm256_shuffle_epi8(byte2, index_bytes);
  const __m256i values3 = _mm256_shuffle_epi8(byte3, index_bytes);
  const __m256i words01 = _mm256_unpacklo_epi8(values0, values1);
  const __m256i words23 = _mm256_unpacklo_epi8(values2, values3);
  const __m256i lanes03 = _mm256_unpacklo_epi16(words01, words23);
  const __m256i lanes47 = _mm256_unpackhi_epi16(words01, words23);
  return q4_lookup16_result{
      .low = _mm256_castsi256_ps(
          _mm256_permute2x128_si256(lanes03, lanes47, 0x20)),
      .high = _mm256_castsi256_ps(
          _mm256_permute2x128_si256(lanes03, lanes47, 0x31))};
}

EMEL_KERNEL_CQ_AVX2_TARGET inline __m256
lookup_codebook8_pshufb(const __m128i index_bytes, const __m128i byte0,
                        const __m128i byte1, const __m128i byte2,
                        const __m128i byte3) noexcept {
  const __m128i values0 = _mm_shuffle_epi8(byte0, index_bytes);
  const __m128i values1 = _mm_shuffle_epi8(byte1, index_bytes);
  const __m128i values2 = _mm_shuffle_epi8(byte2, index_bytes);
  const __m128i values3 = _mm_shuffle_epi8(byte3, index_bytes);
  const __m128i words01 = _mm_unpacklo_epi8(values0, values1);
  const __m128i words23 = _mm_unpacklo_epi8(values2, values3);
  const __m128i low = _mm_unpacklo_epi16(words01, words23);
  const __m128i high = _mm_unpackhi_epi16(words01, words23);
  return _mm256_castsi256_ps(
      _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1));
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void q4_codebook_byte_tables(
    const event::prepared_codebook_q4 &codebook, __m256i &byte0,
    __m256i &byte1, __m256i &byte2, __m256i &byte3) noexcept {
  byte0 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(codebook.byte_planes[0].data()));
  byte1 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(codebook.byte_planes[1].data()));
  byte2 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(codebook.byte_planes[2].data()));
  byte3 = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(codebook.byte_planes[3].data()));
}

EMEL_KERNEL_CQ_AVX2_TARGET inline __m256i
load_selector16(const uint8_t *selectors) noexcept {
  const __m128i low =
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(selectors));
  const __m128i high =
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(selectors + 8u));
  return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
}
EMEL_KERNEL_CQ_AVX2_TARGET inline __m256i
load_selector32(const uint8_t *selectors) noexcept {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(selectors));
}
#endif

EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_avx2_dot_loaded(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht, const uint32_t row_begin,
    const uint32_t row_count, const std::span<float> output,
    const __m256i codebook_byte0, const __m256i codebook_byte1,
    const __m256i codebook_byte2, const __m256i codebook_byte3) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const uint32_t groups_per_row = view.in_pad / view.group;
  for (uint32_t row = 0u; row < row_count; ++row) {
    const uint32_t source_row = row_begin + row;
    const uint8_t *indices =
        view.indices.data() + static_cast<size_t>(source_row) * view.in_pad;
    const float *norms =
        view.norms.data() + static_cast<size_t>(source_row) * groups_per_row;
    // Four independent accumulation chains keep the shuffle/FMA pipeline full.
    // Each group is reduced before its exact decoded fp16 norm is applied, so
    // the hot selector loop performs lookup * activation only.
    float row_result = 0.0f;
    for (uint32_t begin = 0u, group_index = 0u; begin < view.in_pad;
         begin += view.group, ++group_index) {
      __m256 group_accum0 = _mm256_setzero_ps();
      __m256 group_accum1 = _mm256_setzero_ps();
      __m256 group_accum2 = _mm256_setzero_ps();
      __m256 group_accum3 = _mm256_setzero_ps();
      float scalar_tail = 0.0f;
      uint32_t i = 0u;
      for (; i + 64u <= view.group; i += 64u) {
        const uint8_t *chunk = indices + begin + i;
        const float *activation = activation_fwht.data() + begin + i;
        const q4_lookup32_result values0 = lookup_codebook32_pshufb(
            load_selector32(chunk), codebook_byte0, codebook_byte1,
            codebook_byte2, codebook_byte3);
        const q4_lookup32_result values1 = lookup_codebook32_pshufb(
            load_selector32(chunk + 32u), codebook_byte0, codebook_byte1,
            codebook_byte2, codebook_byte3);
        group_accum0 = _mm256_fmadd_ps(values0.values0,
                                       _mm256_loadu_ps(activation),
                                       group_accum0);
        group_accum1 = _mm256_fmadd_ps(values0.values1,
                                       _mm256_loadu_ps(activation + 8u),
                                       group_accum1);
        group_accum2 = _mm256_fmadd_ps(values0.values2,
                                       _mm256_loadu_ps(activation + 16u),
                                       group_accum2);
        group_accum3 = _mm256_fmadd_ps(values0.values3,
                                       _mm256_loadu_ps(activation + 24u),
                                       group_accum3);
        group_accum0 = _mm256_fmadd_ps(values1.values0,
                                       _mm256_loadu_ps(activation + 32u),
                                       group_accum0);
        group_accum1 = _mm256_fmadd_ps(values1.values1,
                                       _mm256_loadu_ps(activation + 40u),
                                       group_accum1);
        group_accum2 = _mm256_fmadd_ps(values1.values2,
                                       _mm256_loadu_ps(activation + 48u),
                                       group_accum2);
        group_accum3 = _mm256_fmadd_ps(values1.values3,
                                       _mm256_loadu_ps(activation + 56u),
                                       group_accum3);
      }
      for (; i + 32u <= view.group; i += 32u) {
        const q4_lookup32_result values = lookup_codebook32_pshufb(
            load_selector32(indices + begin + i), codebook_byte0,
            codebook_byte1, codebook_byte2, codebook_byte3);
        const float *activation = activation_fwht.data() + begin + i;
        group_accum0 = _mm256_fmadd_ps(
            values.values0, _mm256_loadu_ps(activation), group_accum0);
        group_accum1 = _mm256_fmadd_ps(
            values.values1, _mm256_loadu_ps(activation + 8u), group_accum1);
        group_accum2 = _mm256_fmadd_ps(
            values.values2, _mm256_loadu_ps(activation + 16u), group_accum2);
        group_accum3 = _mm256_fmadd_ps(
            values.values3, _mm256_loadu_ps(activation + 24u), group_accum3);
      }
      if (i + 16u <= view.group) {
        const q4_lookup16_result values = lookup_codebook16_pshufb(
            load_selector16(indices + begin + i), codebook_byte0,
            codebook_byte1, codebook_byte2, codebook_byte3);
        group_accum0 = _mm256_fmadd_ps(
            values.low, _mm256_loadu_ps(activation_fwht.data() + begin + i),
            group_accum0);
        group_accum1 = _mm256_fmadd_ps(
            values.high,
            _mm256_loadu_ps(activation_fwht.data() + begin + i + 8u),
            group_accum1);
        i += 16u;
      }
      if (i + 8u <= view.group) {
        const __m128i selectors = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(indices + begin + i));
        const __m256 values = lookup_codebook8_pshufb(
            selectors, _mm256_castsi256_si128(codebook_byte0),
            _mm256_castsi256_si128(codebook_byte1),
            _mm256_castsi256_si128(codebook_byte2),
            _mm256_castsi256_si128(codebook_byte3));
        group_accum0 = _mm256_fmadd_ps(
            values, _mm256_loadu_ps(activation_fwht.data() + begin + i),
            group_accum0);
        i += 8u;
      }
      for (; i < view.group; ++i)
        scalar_tail += detail::code_value<4u>(
                           indices[begin + i], view.group, codebook.values) *
                       activation_fwht[begin + i];
      const __m256 group_accum = _mm256_add_ps(
          _mm256_add_ps(group_accum0, group_accum1),
          _mm256_add_ps(group_accum2, group_accum3));
      alignas(32) float lanes[8];
      _mm256_store_ps(lanes, group_accum);
      const float group_sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                              lanes[4] + lanes[5] + lanes[6] + lanes[7] +
                              scalar_tail;
      row_result += group_sum * norms[group_index];
    }
    output[row] = row_result;
  }
#else
  (void)view;
  (void)codebook;
  (void)activation_fwht;
  (void)row_begin;
  (void)row_count;
  (void)output;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_avx2_dot(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht, const uint32_t row_begin,
    const uint32_t row_count, const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  __m256i codebook_byte0;
  __m256i codebook_byte1;
  __m256i codebook_byte2;
  __m256i codebook_byte3;
  q4_codebook_byte_tables(codebook, codebook_byte0, codebook_byte1,
                          codebook_byte2, codebook_byte3);
  execute_prepared_avx2_dot_loaded(
      view, codebook, activation_fwht, row_begin, row_count, output,
      codebook_byte0, codebook_byte1, codebook_byte2, codebook_byte3);
#else
  (void)view;
  (void)codebook;
  (void)activation_fwht;
  (void)row_begin;
  (void)row_count;
  (void)output;
#endif
}

template <uint32_t Rows>
EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_avx2_dot_row_block(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht, const uint32_t row_begin,
    const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  static_assert(Rows == 4u || Rows == 8u);
  __m256i codebook_byte0;
  __m256i codebook_byte1;
  __m256i codebook_byte2;
  __m256i codebook_byte3;
  q4_codebook_byte_tables(codebook, codebook_byte0, codebook_byte1,
                          codebook_byte2, codebook_byte3);
  const uint32_t groups_per_row = view.in_pad / view.group;
  const uint8_t *row_indices[Rows];
  const float *row_norms[Rows];
  __m256 accum[Rows];
  float scalar_tail[Rows]{};
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
    uint32_t i = 0u;
    for (; i + 8u <= view.group; i += 8u) {
      const __m256 activation =
          _mm256_loadu_ps(activation_fwht.data() + begin + i);
      for (uint32_t row = 0u; row < Rows; ++row) {
        const __m128i index_bytes = _mm_loadl_epi64(
            reinterpret_cast<const __m128i *>(row_indices[row] + begin + i));
        const __m256 values = lookup_codebook8_pshufb(
            index_bytes, _mm256_castsi256_si128(codebook_byte0),
            _mm256_castsi256_si128(codebook_byte1),
            _mm256_castsi256_si128(codebook_byte2),
            _mm256_castsi256_si128(codebook_byte3));
        accum[row] = _mm256_fmadd_ps(_mm256_mul_ps(values, norm[row]),
                                     activation, accum[row]);
      }
    }
    for (; i < view.group; ++i)
      for (uint32_t row = 0u; row < Rows; ++row)
        scalar_tail[row] +=
            detail::code_value<4u>(row_indices[row][begin + i], view.group,
                                   codebook.values) *
            row_norms[row][group_index] * activation_fwht[begin + i];
  }
  for (uint32_t row = 0u; row < Rows; ++row) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accum[row]);
    output[row] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                  lanes[5] + lanes[6] + lanes[7] + scalar_tail[row];
  }
#else
  (void)view;
  (void)codebook;
  (void)activation_fwht;
  (void)row_begin;
  (void)output;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot_block32_loaded(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht,
    const std::span<float> output, const __m256i codebook_byte0,
    const __m256i codebook_byte1, const __m256i codebook_byte2,
    const __m256i codebook_byte3) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  const uint32_t blocked_rows = view.out / 32u * 32u;
  const uint32_t groups_per_row = view.in_pad / view.group;
  for (uint32_t row = 0u; row < blocked_rows; row += 32u) {
    const uint8_t *selectors =
        view.indices_by_input32.data() + static_cast<size_t>(row) * view.in_pad;
    const float *group_norms = view.norms_by_group32.data() +
                               static_cast<size_t>(row) * groups_per_row;
    __m256 row_total0 = _mm256_setzero_ps();
    __m256 row_total1 = _mm256_setzero_ps();
    __m256 row_total2 = _mm256_setzero_ps();
    __m256 row_total3 = _mm256_setzero_ps();
    for (uint32_t begin = 0u, group_index = 0u; begin < view.in_pad;
         begin += view.group, ++group_index) {
      __m256 group_accum0 = _mm256_setzero_ps();
      __m256 group_accum1 = _mm256_setzero_ps();
      __m256 group_accum2 = _mm256_setzero_ps();
      __m256 group_accum3 = _mm256_setzero_ps();
      for (uint32_t i = 0u; i < view.group; ++i) {
        const q4_lookup32_result values = lookup_codebook32_raw(
            load_selector32(selectors + static_cast<size_t>(begin + i) * 32u),
            codebook_byte0, codebook_byte1, codebook_byte2, codebook_byte3);
        const __m256 activation =
            _mm256_set1_ps(activation_fwht[begin + i]);
        group_accum0 =
            _mm256_fmadd_ps(values.values0, activation, group_accum0);
        group_accum1 =
            _mm256_fmadd_ps(values.values1, activation, group_accum1);
        group_accum2 =
            _mm256_fmadd_ps(values.values2, activation, group_accum2);
        group_accum3 =
            _mm256_fmadd_ps(values.values3, activation, group_accum3);
      }
      const float *norms = group_norms + static_cast<size_t>(group_index) * 32u;
      row_total0 = _mm256_fmadd_ps(group_accum0, _mm256_loadu_ps(norms),
                                   row_total0);
      row_total1 = _mm256_fmadd_ps(group_accum1, _mm256_loadu_ps(norms + 8u),
                                   row_total1);
      row_total2 = _mm256_fmadd_ps(group_accum2, _mm256_loadu_ps(norms + 16u),
                                   row_total2);
      row_total3 = _mm256_fmadd_ps(group_accum3, _mm256_loadu_ps(norms + 24u),
                                   row_total3);
    }
    _mm_storeu_ps(output.data() + row, _mm256_castps256_ps128(row_total0));
    _mm_storeu_ps(output.data() + row + 4u,
                  _mm256_castps256_ps128(row_total1));
    _mm_storeu_ps(output.data() + row + 8u,
                  _mm256_castps256_ps128(row_total2));
    _mm_storeu_ps(output.data() + row + 12u,
                  _mm256_castps256_ps128(row_total3));
    _mm_storeu_ps(output.data() + row + 16u,
                  _mm256_extractf128_ps(row_total0, 1));
    _mm_storeu_ps(output.data() + row + 20u,
                  _mm256_extractf128_ps(row_total1, 1));
    _mm_storeu_ps(output.data() + row + 24u,
                  _mm256_extractf128_ps(row_total2, 1));
    _mm_storeu_ps(output.data() + row + 28u,
                  _mm256_extractf128_ps(row_total3, 1));
  }
  if (blocked_rows < view.out)
    execute_prepared_avx2_dot_loaded(
        view, codebook, activation_fwht, blocked_rows,
        view.out - blocked_rows, output.subspan(blocked_rows), codebook_byte0,
        codebook_byte1, codebook_byte2, codebook_byte3);
#else
  (void)view;
  (void)codebook;
  (void)activation_fwht;
  (void)output;
  (void)codebook_byte0;
  (void)codebook_byte1;
  (void)codebook_byte2;
  (void)codebook_byte3;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void execute_prepared_avx2_dot_block32(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht,
    const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  __m256i codebook_byte0;
  __m256i codebook_byte1;
  __m256i codebook_byte2;
  __m256i codebook_byte3;
  q4_codebook_byte_tables(codebook, codebook_byte0, codebook_byte1,
                          codebook_byte2, codebook_byte3);
  execute_prepared_avx2_dot_block32_loaded(
      view, codebook, activation_fwht, output, codebook_byte0, codebook_byte1,
      codebook_byte2, codebook_byte3);
#else
  (void)view;
  (void)codebook;
  (void)activation_fwht;
  (void)output;
#endif
}


template <uint32_t Rows>
EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot_blocked(const event::prepared_q4_view &view,
                                  const event::prepared_codebook_q4 &codebook,
                                  const std::span<const float> activation_fwht,
                                  const std::span<float> output) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  static_assert(Rows == 4u || Rows == 8u);
  const uint32_t blocked_rows = view.out / Rows * Rows;
  for (uint32_t row = 0u; row < blocked_rows; row += Rows)
    execute_prepared_avx2_dot_row_block<Rows>(
        view, codebook, activation_fwht, row, output.subspan(row, Rows));
  if (blocked_rows < view.out)
    execute_prepared_avx2_dot(view, codebook, activation_fwht, blocked_rows,
                              view.out - blocked_rows,
                              output.subspan(blocked_rows));
#else
  (void)view;
  (void)codebook;
  (void)activation_fwht;
  (void)output;
#endif
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot_blocked4(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht,
    const std::span<float> output) noexcept {
  execute_prepared_avx2_dot_blocked<4u>(view, codebook, activation_fwht, output);
}

EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_dot_blocked8(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook,
    const std::span<const float> activation_fwht,
    const std::span<float> output) noexcept {
  execute_prepared_avx2_dot_blocked<8u>(view, codebook, activation_fwht, output);
}
template <bool Rows>
EMEL_KERNEL_CQ_AVX2_TARGET inline void
execute_prepared_avx2_gemv(const event::prepared_q4_view &view,
                           const event::prepared_codebook_q4 &codebook,
                           const std::span<const float> activation,
                           const uint32_t row_begin, const uint32_t row_count,
                           const std::span<float> output,
                           const std::span<float> workspace) noexcept {
  detail::compute_fwht_groups(activation, view.in, view.group,
                              workspace.first(view.in_pad));
  if constexpr (Rows)
    execute_prepared_avx2_dot(view, codebook, workspace.first(view.in_pad),
                              row_begin, row_count, output);
  else
    execute_prepared_avx2_dot_block32(view, codebook,
                                      workspace.first(view.in_pad), output);
}

inline void execute_prepared_dequant_rows(
    const event::prepared_dequant_rows_request &request) noexcept {
  const auto &view = request.weights;
  const float *codebook = detail::codebook_for<4u>(request.codebook.values);
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
  __m256 q4_codebook_low{};
  __m256 q4_codebook_high{};
  __m128i q4_duplicate_mask{};
  __m128i q4_high_nibble_mask{};
  __m128i q4_nibble_mask{};
  if constexpr (Bits == 4u) {
    q4_codebook_low = _mm256_loadu_ps(codebook);
    q4_codebook_high = _mm256_loadu_ps(codebook + 8u);
    q4_duplicate_mask = _mm_setr_epi8(
        0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    q4_high_nibble_mask =
        _mm_setr_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0);
    q4_nibble_mask = _mm_set1_epi8(0x0f);
  }
  for (uint32_t row = 0u; row < out; ++row) {
    const uint8_t *row_packed = base + static_cast<size_t>(row) * packed_row;
    const uint8_t *row_norms = norms + static_cast<size_t>(row) * norm_row;
    float row_result = 0.0f;
    for (uint32_t begin = 0u, group_index = 0u; begin < in_pad;
         begin += group, ++group_index) {
      const float norm =
          detail::fp16_to_fp32(detail::load_u16(row_norms + group_index * 2u));
      const uint8_t *group_packed = row_packed + group_index * group_bytes;
      __m256 group_accum0 = _mm256_setzero_ps();
      __m256 group_accum1 = _mm256_setzero_ps();
      float scalar_tail = 0.0f;
      uint32_t i = 0u;
      for (; i + 16u <= group; i += 16u) {
        const __m256 activation0 =
            _mm256_loadu_ps(request.workspace.data() + begin + i);
        const __m256 activation1 =
            _mm256_loadu_ps(request.workspace.data() + begin + i + 8u);
        __m256 values0;
        __m256 values1;
        if constexpr (Bits == 4u) {
          uint64_t packed_word = 0u;
          std::memcpy(&packed_word, group_packed + (i >> 1u),
                      sizeof(packed_word));
          const __m128i packed_v =
              _mm_cvtsi64_si128(static_cast<int64_t>(packed_word));
          const auto decode_values = [&](const __m128i packed4) {
            const __m128i duplicated =
                _mm_shuffle_epi8(packed4, q4_duplicate_mask);
            const __m128i low_nibbles =
                _mm_and_si128(duplicated, q4_nibble_mask);
            const __m128i high_nibbles = _mm_and_si128(
                _mm_srli_epi16(duplicated, 4), q4_nibble_mask);
            const __m128i indices_u8 = _mm_blendv_epi8(
                low_nibbles, high_nibbles, q4_high_nibble_mask);
            const __m256i index_v = _mm256_cvtepu8_epi32(indices_u8);
            const __m256i table_index =
                _mm256_and_si256(index_v, _mm256_set1_epi32(7));
            const __m256 low_values =
                _mm256_permutevar8x32_ps(q4_codebook_low, table_index);
            const __m256 high_values =
                _mm256_permutevar8x32_ps(q4_codebook_high, table_index);
            const __m256 high_table_mask =
                _mm256_castsi256_ps(_mm256_slli_epi32(index_v, 28));
            return _mm256_blendv_ps(low_values, high_values, high_table_mask);
          };
          values0 = decode_values(packed_v);
          values1 = decode_values(_mm_srli_si128(packed_v, 4));
        } else {
          alignas(32) int32_t indices0[8];
          alignas(32) int32_t indices1[8];
          for (uint32_t lane = 0u; lane < 8u; ++lane) {
            indices0[lane] = static_cast<int32_t>(
                detail::unpack_index<Bits>(group_packed, i + lane));
            indices1[lane] = static_cast<int32_t>(
                detail::unpack_index<Bits>(group_packed, i + 8u + lane));
          }
          values0 = _mm256_i32gather_ps(
              codebook,
              _mm256_load_si256(reinterpret_cast<const __m256i *>(indices0)),
              4);
          values1 = _mm256_i32gather_ps(
              codebook,
              _mm256_load_si256(reinterpret_cast<const __m256i *>(indices1)),
              4);
        }
        group_accum0 = _mm256_fmadd_ps(values0, activation0, group_accum0);
        group_accum1 = _mm256_fmadd_ps(values1, activation1, group_accum1);
      }
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
          const __m128i low_nibbles =
              _mm_and_si128(duplicated, q4_nibble_mask);
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
          values = _mm256_i32gather_ps(
              codebook,
              _mm256_load_si256(reinterpret_cast<const __m256i *>(indices)),
              4);
        }
        group_accum0 = _mm256_fmadd_ps(values, activation, group_accum0);
      }
      for (; i < group; ++i) {
        const uint32_t index = detail::unpack_index<Bits>(group_packed, i);
        scalar_tail += detail::code_value<Bits>(index, group, request.codebook) *
                       request.workspace[begin + i];
      }
      const __m256 group_accum = _mm256_add_ps(group_accum0, group_accum1);
      alignas(32) float lanes[8];
      _mm256_store_ps(lanes, group_accum);
      const float group_sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                              lanes[4] + lanes[5] + lanes[6] + lanes[7] +
                              scalar_tail;
      row_result += group_sum * norm;
    }
    request.output[row] = row_result * request.output_scale;
  }
#else
  execute_scalar_gemv<Bits>(request);
#endif
}
inline uint64_t timing_now(const context &ctx) noexcept {
  return ctx.timing_enabled && ctx.timing_now != nullptr ? ctx.timing_now()
                                                          : 0u;
}

inline void scale_output(const std::span<float> output,
                         const float scale) noexcept {
  for (float &value : output)
    value *= scale;
}

struct effect_quantize_a8 {
  void operator()(const event::quantize_a8 &ev, context &ctx) const noexcept {
    const uint64_t begin = timing_now(ctx);
    quantize_a8(ev.request);
    if (ctx.timing_enabled)
      ctx.timing.quantize_nanoseconds += timing_now(ctx) - begin;
    ev.result.accepted = true;
    ++ctx.quantize_calls;
  }
};

struct effect_execute_fwht_avx2 {
  void operator()(const event::execute_fwht_avx2 &ev,
                  context &ctx) const noexcept {
    const uint64_t begin = timing_now(ctx);
    detail::fwht128_avx2(ev.request.values.data());
    if (ctx.timing_enabled)
      ctx.timing.fwht_nanoseconds += timing_now(ctx) - begin;
    ev.result.accepted = true;
  }
};

struct effect_prepare_codebook_q4 {
  void operator()(const event::prepare_codebook_q4 &ev,
                  context &) const noexcept {
    prepare_codebook_q4(ev.request);
    ev.result.accepted = true;
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
    const uint64_t fwht_begin = timing_now(ctx);
    if constexpr (true)
      detail::compute_fwht128_groups_avx2(
          ev.request.activation, ev.request.weights.in,
          ev.request.workspace.first(ev.request.weights.in_pad));
    if (ctx.timing_enabled)
      ctx.timing.fwht_nanoseconds += timing_now(ctx) - fwht_begin;
    const uint64_t dot_begin = timing_now(ctx);
    execute_prepared_avx2_dot_block32(
        ev.request.weights, ev.request.codebook,
        ev.request.workspace.first(ev.request.weights.in_pad),
        ev.request.output);
    scale_output(ev.request.output.first(ev.request.weights.out),
                 ev.request.output_scale);
    if (ctx.timing_enabled)
      ctx.timing.dot_full_nanoseconds += timing_now(ctx) - dot_begin;
    ev.result.accepted = true;
    ++ctx.prepared_calls;
  }
};

struct effect_execute_prepared_avx2_batch4_q4 {
  void operator()(const event::execute_prepared_avx2_batch4_q4 &ev,
                  context &ctx) const noexcept {
    const auto &request = ev.request;
    const auto &first = *request.targets[0].weights;
    const uint64_t fwht_begin = timing_now(ctx);
    if constexpr (true)
      detail::compute_fwht128_groups_avx2(
          request.activation, first.in, request.workspace.first(first.in_pad));
    if (ctx.timing_enabled)
      ctx.timing.fwht_nanoseconds += timing_now(ctx) - fwht_begin;
    const uint64_t dot_begin = timing_now(ctx);
    __m256i codebook_byte0;
    __m256i codebook_byte1;
    __m256i codebook_byte2;
    __m256i codebook_byte3;
    q4_codebook_byte_tables(request.codebook, codebook_byte0, codebook_byte1,
                            codebook_byte2, codebook_byte3);
    for (const auto &target : request.targets) {
      execute_prepared_avx2_dot_block32_loaded(
          *target.weights, request.codebook,
          request.workspace.first(first.in_pad), target.output, codebook_byte0,
          codebook_byte1, codebook_byte2, codebook_byte3);
      scale_output(target.output.first(target.weights->out),
                   request.output_scale);
    }
    if (ctx.timing_enabled)
      ctx.timing.dot_batch_nanoseconds += timing_now(ctx) - dot_begin;
    ev.result.accepted = true;
    ctx.prepared_calls += request.targets.size();
  }
};

struct effect_execute_prepared_avx2_rows_q4 {
  void operator()(const event::execute_prepared_avx2_rows_q4 &ev,
                  context &ctx) const noexcept {
    const uint64_t fwht_begin = timing_now(ctx);
    if constexpr (true)
      detail::compute_fwht128_groups_avx2(
          ev.request.activation, ev.request.weights.in,
          ev.request.workspace.first(ev.request.weights.in_pad));
    if (ctx.timing_enabled)
      ctx.timing.fwht_nanoseconds += timing_now(ctx) - fwht_begin;
    const uint64_t dot_begin = timing_now(ctx);
    execute_prepared_avx2_dot(
        ev.request.weights, ev.request.codebook,
        ev.request.workspace.first(ev.request.weights.in_pad),
        ev.request.row_begin, ev.request.row_count, ev.request.output);
    scale_output(ev.request.output.first(ev.request.row_count),
                 ev.request.output_scale);
    if (ctx.timing_enabled)
      ctx.timing.dot_rows_nanoseconds += timing_now(ctx) - dot_begin;
    ev.result.accepted = true;
    ++ctx.prepared_calls;
  }
};
struct effect_execute_prepared_dequant_q4 {
  void operator()(const event::execute_prepared_dequant_q4 &ev,
                  context &ctx) const noexcept {
    const uint64_t begin = timing_now(ctx);
    execute_prepared_dequant_rows(ev.request);
    if (ctx.timing_enabled)
      ctx.timing.dequant_nanoseconds += timing_now(ctx) - begin;
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
struct effect_configure_timing {
  void operator()(const event::configure_timing &ev,
                  context &ctx) const noexcept {
    if (ev.enabled && !ctx.timing_enabled)
      ctx.timing = {};
    ctx.timing_enabled = ev.enabled;
    ctx.timing_now = ev.now;
  }
};

struct effect_capture_timing {
  void operator()(const event::capture_timing &ev,
                  const context &ctx) const noexcept {
    ev.breakdown = ctx.timing;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::cq::action
