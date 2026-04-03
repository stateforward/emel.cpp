#pragma once

#include <cstdint>
#include <initializer_list>
#include <unordered_set>
#include <utility>

namespace emel::text {

struct range_nfd {
  uint32_t first;
  uint32_t last;
  uint32_t nfd;
};

inline constexpr uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;

// Lists stay in ascending order to enable binary search.
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;

}  // namespace emel::text
