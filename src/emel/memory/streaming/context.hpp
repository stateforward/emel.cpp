#pragma once

#include <cstdint>

namespace emel::memory::streaming {

struct dependencies {
  int32_t capacity;
};

namespace action {

struct context {
  explicit context(const dependencies &deps) noexcept
      : capacity(deps.capacity) {}

  // Persistent source state is deliberately minimal. Fill/full is represented
  // by SML state, and visible-window bounds are derived into event outputs.
  const int32_t capacity;
  int64_t next_logical_position = 0;

  // This is derivable as next_logical_position % capacity, but retaining the
  // cursor removes a 64-bit remainder from every hot streaming advance. The
  // transition actions maintain the equivalence invariant.
  int32_t next_physical_position = 0;
};

} // namespace action

} // namespace emel::memory::streaming
