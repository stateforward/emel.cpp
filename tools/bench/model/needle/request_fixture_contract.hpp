#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace emel::bench::needle_request {

inline bool token_ids_match(const std::span<const int32_t> expected,
                            const std::span<const int32_t> actual) noexcept {
  return expected.size() == actual.size() &&
         std::equal(expected.begin(), expected.end(), actual.begin());
}

} // namespace emel::bench::needle_request
