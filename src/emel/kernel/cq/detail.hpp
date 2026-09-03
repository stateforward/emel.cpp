#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "emel/cact/loader/events.hpp"
#include "emel/kernel/detail.hpp"

namespace emel::kernel::cq::detail {

inline constexpr uint32_t k_ternary_record_bits = 5u;
inline constexpr float k_ternary_centroid = 1.2240064f;
inline constexpr uint32_t k_max_group = 4096u;
inline float round_ties_to_even(const float value) noexcept {
  const int32_t truncated = static_cast<int32_t>(value);
  const float truncated_value = static_cast<float>(truncated);
  const float fraction = value - truncated_value;
  if (fraction > 0.5f || (fraction == 0.5f && (truncated & 1) != 0))
    return static_cast<float>(truncated + 1);
  if (fraction < -0.5f || (fraction == -0.5f && (truncated & 1) != 0))
    return static_cast<float>(truncated - 1);
  return truncated_value;
}

struct layout {
  uint32_t in_pad = 0u;
  uint64_t packed_row_bytes = 0u;
  uint64_t packed_bytes = 0u;
  uint64_t norm_row_bytes = 0u;
  uint64_t norm_bytes = 0u;
  uint64_t total_bytes = 0u;
};

inline bool checked_multiply_u64(const uint64_t lhs, const uint64_t rhs,
                                 uint64_t &product) noexcept {
  if (lhs != 0u && rhs > UINT64_MAX / lhs)
    return false;
  product = lhs * rhs;
  return true;
}

inline bool checked_add_u64(const uint64_t lhs, const uint64_t rhs,
                            uint64_t &sum) noexcept {
  if (rhs > UINT64_MAX - lhs)
    return false;
  sum = lhs + rhs;
  return true;
}

template <uint32_t Bits>
inline bool checked_layout(const uint32_t out, const uint32_t in,
                           const uint32_t group, layout &result) noexcept {
  constexpr uint32_t storage_bits = Bits == k_ternary_record_bits ? 2u : Bits;
  if (out == 0u || in == 0u || group == 0u)
    return false;
  const uint64_t in_pad =
      (static_cast<uint64_t>(in) + group - 1u) / group * group;
  if (in_pad > UINT32_MAX)
    return false;
  uint64_t packed_bits = 0u;
  uint64_t packed_bytes = 0u;
  uint64_t norm_row_bytes = 0u;
  uint64_t norm_bytes = 0u;
  uint64_t total_bytes = 0u;
  if (!checked_multiply_u64(in_pad, storage_bits, packed_bits) ||
      (packed_bits & 7u) != 0u ||
      !checked_multiply_u64(out, packed_bits / 8u, packed_bytes) ||
      !checked_multiply_u64(in_pad / group, 2u, norm_row_bytes) ||
      !checked_multiply_u64(out, norm_row_bytes, norm_bytes) ||
      !checked_add_u64(packed_bytes, norm_bytes, total_bytes))
    return false;
  result = layout{.in_pad = static_cast<uint32_t>(in_pad),
                  .packed_row_bytes = packed_bits / 8u,
                  .packed_bytes = packed_bytes,
                  .norm_row_bytes = norm_row_bytes,
                  .norm_bytes = norm_bytes,
                  .total_bytes = total_bytes};
  return true;
}
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET
#endif
#else
#define EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET
#endif

inline float fp16_to_fp32(const uint16_t bits) noexcept {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
  const uint32_t exp = (bits >> 10u) & 0x1Fu;
  const uint32_t mant = bits & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u)
      return std::bit_cast<float>(sign);
    float value = static_cast<float>(mant) * 0x1.0p-24f;
    return sign != 0u ? -value : value;
  }
  if (exp == 31u)
    return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13u));
  return std::bit_cast<float>(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

inline uint16_t load_u16(const uint8_t *p) noexcept {
  return static_cast<uint16_t>(p[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8u);
}

template <uint32_t Bits>
inline uint32_t unpack_index(const uint8_t *packed,
                             const size_t index) noexcept {
  if constexpr (Bits == k_ternary_record_bits) {
    const uint8_t crumb =
        static_cast<uint8_t>((packed[index >> 2u] >> ((index & 3u) * 2u)) & 3u);
    return crumb == 3u ? 0u : static_cast<uint32_t>(crumb) + 1u;
  } else {
    constexpr uint32_t mask = (1u << Bits) - 1u;
    const size_t bit = index * Bits;
    const size_t byte = bit >> 3u;
    const uint32_t shift = static_cast<uint32_t>(bit & 7u);
    uint32_t word = packed[byte];
    if constexpr (Bits > 1u) {
      if (shift + Bits > 8u)
        word |= static_cast<uint32_t>(packed[byte + 1u]) << 8u;
    }
    if constexpr (Bits > 4u) {
      if (shift + Bits > 16u)
        word |= static_cast<uint32_t>(packed[byte + 2u]) << 16u;
    }
    return (word >> shift) & mask;
  }
}

template <uint32_t Bits>
inline const float *
codebook_for(const std::span<const float> codebook) noexcept {
  static_assert(Bits == 2u || Bits == 3u || Bits == 4u);
  if constexpr (Bits == 2u)
    return codebook.data();
  if constexpr (Bits == 3u)
    return codebook.data() + 4u;
  return codebook.data() + 12u;
}

// CQ4 sign/rank lookup is lossless only when every negative level has an
// exact positive counterpart. Match by value rather than index ordering so
// callers may use any codebook permutation. The bitwise sign relation is the
// strongest accepted form; exact numeric negation also admits signed zero.
inline bool
q4_codebook_is_symmetric(const std::span<const float> codebook) noexcept {
  if (codebook.size() < 28u)
    return false;
  const float *levels = codebook_for<4u>(codebook);
  for (uint32_t i = 0u; i < 16u; ++i) {
    const uint32_t bits = std::bit_cast<uint32_t>(levels[i]);
    bool found = false;
    for (uint32_t j = 0u; j < 16u; ++j) {
      if (j == i)
        continue;
      const uint32_t other_bits = std::bit_cast<uint32_t>(levels[j]);
      if (other_bits == (bits ^ 0x80000000u) || levels[j] == -levels[i]) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

inline void fwht(float *values, const uint32_t n) noexcept {
  emel::kernel::detail::fwht_normalized(values, n);
}
EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET inline void
fwht128_avx2(float *values) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  for (uint32_t base = 0u; base < 128u; base += 8u) {
    const __m256 x = _mm256_loadu_ps(values + base);
    const __m256 swapped1 = _mm256_permute_ps(x, 0xb1);
    const __m256 sums1 = _mm256_add_ps(x, swapped1);
    const __m256 diffs1 = _mm256_mul_ps(
        _mm256_sub_ps(x, swapped1),
        _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f));
    const __m256 stage1 = _mm256_blend_ps(sums1, diffs1, 0xaau);
    const __m256 swapped2 = _mm256_permute_ps(stage1, 0x4e);
    const __m256 sums2 = _mm256_add_ps(stage1, swapped2);
    const __m256 diffs2 = _mm256_mul_ps(
        _mm256_sub_ps(stage1, swapped2),
        _mm256_setr_ps(1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f));
    const __m256 stage2 = _mm256_blend_ps(sums2, diffs2, 0xccu);
    const __m256 swapped4 = _mm256_permute2f128_ps(stage2, stage2, 0x01);
    const __m256 sums4 = _mm256_add_ps(stage2, swapped4);
    const __m256 diffs4 = _mm256_mul_ps(
        _mm256_sub_ps(stage2, swapped4),
        _mm256_setr_ps(1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f));
    _mm256_storeu_ps(values + base, _mm256_blend_ps(sums4, diffs4, 0xf0u));
  }
  for (uint32_t step = 8u; step < 128u; step <<= 1u)
    for (uint32_t base = 0u; base < 128u; base += step << 1u)
      for (uint32_t j = 0u; j < step; j += 8u) {
        const __m256 a = _mm256_loadu_ps(values + base + j);
        const __m256 b = _mm256_loadu_ps(values + base + step + j);
        _mm256_storeu_ps(values + base + j, _mm256_add_ps(a, b));
        _mm256_storeu_ps(values + base + step + j, _mm256_sub_ps(a, b));
      }
  const __m256 scale = _mm256_set1_ps(0.08838834764831844055f);
  for (uint32_t i = 0u; i < 128u; i += 8u)
    _mm256_storeu_ps(values + i,
                     _mm256_mul_ps(_mm256_loadu_ps(values + i), scale));
#else
  fwht(values, 128u);
#endif
}

inline bool is_power_of_two(const uint32_t n) noexcept {
  return n != 0u && (n & (n - 1u)) == 0u;
}

template <uint32_t Bits>
inline size_t packed_row_bytes(const uint32_t in_pad) noexcept {
  constexpr uint32_t storage_bits = Bits == k_ternary_record_bits ? 2u : Bits;
  return static_cast<size_t>(in_pad) * storage_bits / 8u;
}

template <uint32_t Bits>
inline bool valid_view(const emel::cact::loader::tensor_view &view,
                       const std::span<const float> codebook,
                       const std::span<const float> activation,
                       const std::span<float> output) noexcept {
  const uint32_t out = view.shape[0];
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  layout checked{};
  return view.data != nullptr && view.bits == Bits && group <= k_max_group &&
         is_power_of_two(group) && activation.size() >= in &&
         output.size() >= out &&
         (Bits == k_ternary_record_bits || codebook.size() >= 28u) &&
         checked_layout<Bits>(out, in, group, checked) &&
         checked.total_bytes <= view.nbytes;
}

template <uint32_t Bits>
inline float code_value(const uint32_t index, const uint32_t group,
                        const std::span<const float> codebook) noexcept {
  if constexpr (Bits == k_ternary_record_bits) {
    const float value = index == 0u   ? -k_ternary_centroid
                        : index == 1u ? 0.0f
                                      : k_ternary_centroid;
    return value / std::sqrt(static_cast<float>(group));
  } else {
    return codebook_for<Bits>(codebook)[index];
  }
}
template <uint32_t Bits>
inline float
dequant_dot_row(const uint8_t *packed, const uint8_t *norms,
                const uint32_t group, const uint32_t in_pad,
                const std::span<const float> codebook,
                const std::span<const float> activation_fwht) noexcept {
  const size_t group_bytes = packed_row_bytes<Bits>(group);
  float result = 0.0f;
  for (uint32_t begin = 0u, group_index = 0u; begin < in_pad;
       begin += group, ++group_index) {
    const float norm = fp16_to_fp32(load_u16(norms + group_index * 2u));
    const uint8_t *group_packed = packed + group_index * group_bytes;
    for (uint32_t i = 0u; i < group; ++i) {
      const uint32_t index = unpack_index<Bits>(group_packed, i);
      result += code_value<Bits>(index, group, codebook) * norm *
                activation_fwht[begin + i];
    }
  }
  return result;
}

template <uint32_t Bits>
inline void dequant_row_values(const uint8_t *packed, const uint8_t *norms,
                               const uint32_t in, const uint32_t group,
                               const uint32_t in_pad,
                               const std::span<const float> codebook,
                               const float scale, float *row_out) noexcept {
  const size_t group_bytes = packed_row_bytes<Bits>(group);
  float values[k_max_group];
  for (uint32_t begin = 0u, group_index = 0u; begin < in_pad;
       begin += group, ++group_index) {
    const float norm = fp16_to_fp32(load_u16(norms + group_index * 2u));
    const uint8_t *group_packed = packed + group_index * group_bytes;
    for (uint32_t i = 0u; i < group; ++i) {
      const uint32_t index = unpack_index<Bits>(group_packed, i);
      values[i] = code_value<Bits>(index, group, codebook) * norm;
    }
    fwht(values, group);
    const uint32_t keep =
        begin + group <= in ? group : (in > begin ? in - begin : 0u);
    for (uint32_t i = 0u; i < keep; ++i)
      row_out[begin + i] = values[i] * scale;
  }
}

// Shared structural validity for row-range requests: packed geometry checks
// identical to `valid_view` minus the activation/output coupling.
template <uint32_t Bits>
inline bool valid_packed_view(const emel::cact::loader::tensor_view &view,
                              const std::span<const float> codebook) noexcept {
  layout checked{};
  return view.data != nullptr && view.bits == Bits &&
         view.group <= k_max_group && is_power_of_two(view.group) &&
         (Bits == k_ternary_record_bits || codebook.size() >= 28u) &&
         checked_layout<Bits>(view.shape[0], view.shape[1], view.group,
                              checked) &&
         checked.total_bytes <= view.nbytes;
}

inline void compute_fwht_groups(const std::span<const float> activation,
                                const uint32_t in, const uint32_t group,
                                const uint32_t in_pad,
                                std::span<float> transformed) noexcept {
  for (uint32_t begin = 0u; begin < in_pad; begin += group) {
    for (uint32_t i = 0u; i < group; ++i)
      transformed[begin + i] = begin + i < in ? activation[begin + i] : 0.0f;
    fwht(transformed.data() + begin, group);
  }
}

EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET inline void
compute_fwht128_groups_avx2(const std::span<const float> activation,
                            const uint32_t in, const uint32_t in_pad,
                            std::span<float> transformed) noexcept {
  for (uint32_t begin = 0u; begin < in_pad; begin += 128u) {
    for (uint32_t i = 0u; i < 128u; ++i)
      transformed[begin + i] = begin + i < in ? activation[begin + i] : 0.0f;
    fwht128_avx2(transformed.data() + begin);
  }
}
EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET inline void
compute_fwht128_groups_avx2(const std::span<const float> activation,
                            const uint32_t in,
                            std::span<float> transformed) noexcept {
  const uint64_t in_pad = (static_cast<uint64_t>(in) + 127u) / 128u * 128u;
  if (in_pad <= UINT32_MAX)
    compute_fwht128_groups_avx2(activation, in, static_cast<uint32_t>(in_pad),
                                transformed);
}

#undef EMEL_KERNEL_CQ_DETAIL_AVX2_TARGET

} // namespace emel::kernel::cq::detail
