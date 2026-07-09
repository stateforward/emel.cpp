#pragma once

#include <cstdint>

#include "emel/speech/generator/moshi/context.hpp"

namespace emel::speech::generator::moshi::detail {

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type &ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

inline int32_t cache_position(const action::lmgen_state &lmgen,
                              const int64_t offset) noexcept {
  const int64_t row_count = lmgen.cache_row_count;
  const int64_t position = offset % row_count;
  return static_cast<int32_t>(position + ((position < 0) * row_count));
}

inline int32_t &cache_at(action::lmgen_state &lmgen, const int32_t row,
                         const int32_t column) noexcept {
  return lmgen
      .cache[static_cast<size_t>(row * action::k_max_codebooks + column)];
}

inline int32_t cache_at(const action::lmgen_state &lmgen, const int32_t row,
                        const int32_t column) noexcept {
  return lmgen
      .cache[static_cast<size_t>(row * action::k_max_codebooks + column)];
}

} // namespace emel::speech::generator::moshi::detail
