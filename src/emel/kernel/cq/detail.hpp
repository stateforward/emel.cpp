#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/cact/loader/events.hpp"

namespace emel::kernel::cq::detail {

inline constexpr uint32_t k_ternary_record_bits = 5u;
inline constexpr float k_ternary_centroid = 1.2240064f;
inline constexpr uint32_t k_max_group = 4096u;

inline float fp16_to_fp32(const uint16_t bits) noexcept {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16u;
  const uint32_t exp = (bits >> 10u) & 0x1Fu;
  const uint32_t mant = bits & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) return std::bit_cast<float>(sign);
    float value = static_cast<float>(mant) * 0x1.0p-24f;
    return sign != 0u ? -value : value;
  }
  if (exp == 31u) {
    return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13u));
  }
  return std::bit_cast<float>(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

inline uint16_t load_u16(const uint8_t *p) noexcept {
  return static_cast<uint16_t>(p[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8u);
}

inline uint32_t packed_index(const uint8_t *packed, const size_t index,
                             const uint32_t bits) noexcept {
  const size_t bit = index * bits;
  const size_t byte = bit >> 3u;
  const uint32_t shift = static_cast<uint32_t>(bit & 7u);
  uint32_t word = packed[byte];
  if (shift + bits > 8u) word |= static_cast<uint32_t>(packed[byte + 1u]) << 8u;
  if (shift + bits > 16u) word |= static_cast<uint32_t>(packed[byte + 2u]) << 16u;
  return (word >> shift) & ((1u << bits) - 1u);
}

inline uint32_t unpack_index(const uint8_t *packed, const size_t index,
                             const uint32_t record_bits) noexcept {
  if (record_bits != k_ternary_record_bits) {
    return packed_index(packed, index, record_bits);
  }
  const uint8_t crumb = static_cast<uint8_t>((packed[index >> 2u] >>
                                               ((index & 3u) * 2u)) & 3u);
  return crumb == 3u ? 0u : static_cast<uint32_t>(crumb) + 1u;
}

inline const float *codebook_for(const std::span<const float> codebook,
                                 const uint32_t bits) noexcept {
  if (bits == 2u) return codebook.data();
  if (bits == 3u) return codebook.data() + 4u;
  return codebook.data() + 12u;
}

inline void fwht(float *values, const uint32_t n) noexcept {
  for (uint32_t step = 1u; step < n; step <<= 1u) {
    for (uint32_t base = 0u; base < n; base += step << 1u) {
      for (uint32_t j = 0u; j < step; ++j) {
        const float a = values[base + j];
        const float b = values[base + step + j];
        values[base + j] = a + b;
        values[base + step + j] = a - b;
      }
    }
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(n));
  for (uint32_t i = 0u; i < n; ++i) values[i] *= scale;
}

inline bool is_power_of_two(const uint32_t n) noexcept {
  return n != 0u && (n & (n - 1u)) == 0u;
}

inline size_t packed_row_bytes(const uint32_t in_pad,
                               const uint32_t record_bits) noexcept {
  return static_cast<size_t>(in_pad) *
         (record_bits == k_ternary_record_bits ? 2u : record_bits) / 8u;
}

inline bool valid_view(const emel::cact::loader::tensor_view &view,
                      const std::span<const float> codebook,
                      const std::span<const float> activation,
                      const std::span<float> output) noexcept {
  const uint32_t out = view.shape[0];
  const uint32_t in = view.shape[1];
  const uint32_t group = view.group;
  if (view.data == nullptr || out == 0u || in == 0u || group == 0u ||
      group > k_max_group || !is_power_of_two(group) || activation.size() < in ||
      output.size() < out || (view.bits != 2u && view.bits != 3u &&
                              view.bits != 4u && view.bits != k_ternary_record_bits) ||
      (view.bits != k_ternary_record_bits && codebook.size() < 28u)) {
    return false;
  }
  const uint64_t in_pad = (static_cast<uint64_t>(in) + group - 1u) / group * group;
  const uint64_t packed = static_cast<uint64_t>(out) * packed_row_bytes(
      static_cast<uint32_t>(in_pad), view.bits);
  const uint64_t norms = static_cast<uint64_t>(out) * (in_pad / group) * 2u;
  return packed + norms <= view.nbytes;
}

inline float ternary_code(const uint32_t index, const uint32_t group) noexcept {
  const float value = index == 0u ? -k_ternary_centroid
                    : index == 1u ? 0.0f
                                   : k_ternary_centroid;
  return value / std::sqrt(static_cast<float>(group));
}

inline float code_value(const uint32_t index, const uint32_t record_bits,
                        const uint32_t group,
                        const std::span<const float> codebook) noexcept {
  return record_bits == k_ternary_record_bits
             ? ternary_code(index, group)
             : codebook_for(codebook, record_bits)[index];
}

inline float dequant_dot_row(const uint8_t *packed, const uint8_t *norms,
                             const uint32_t in, const uint32_t group,
                             const uint32_t record_bits,
                             const std::span<const float> codebook,
                             const std::span<const float> activation_fwht) noexcept {
  const uint32_t in_pad = (in + group - 1u) / group * group;
  const size_t group_bytes = packed_row_bytes(group, record_bits);
  float result = 0.0f;
  for (uint32_t begin = 0u, group_index = 0u; begin < in_pad;
       begin += group, ++group_index) {
    const float norm = fp16_to_fp32(load_u16(norms + group_index * 2u));
    const uint8_t *group_packed = packed + group_index * group_bytes;
    for (uint32_t i = 0u; i < group; ++i) {
      const uint32_t index = unpack_index(group_packed, i, record_bits);
      result += code_value(index, record_bits, group, codebook) * norm *
                activation_fwht[begin + i];
    }
  }
  return result;
}

inline void compute_fwht_groups(const std::span<const float> activation,
                                const uint32_t in, const uint32_t group,
                                std::span<float> transformed) noexcept {
  const uint32_t in_pad = (in + group - 1u) / group * group;
  for (uint32_t begin = 0u; begin < in_pad; begin += group) {
    for (uint32_t i = 0u; i < group; ++i)
      transformed[begin + i] = begin + i < in ? activation[begin + i] : 0.0f;
    fwht(transformed.data() + begin, group);
  }
}

} // namespace emel::kernel::cq::detail
