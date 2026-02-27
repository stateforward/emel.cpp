#pragma once

#include <array>
#include <cstddef>

#include "emel/memory/view.hpp"

namespace emel::memory::detail {

inline void clear_snapshot_noop(view::snapshot &) noexcept {}

inline void clear_snapshot(view::snapshot & snapshot) noexcept {
  snapshot.max_sequences = 0;
  snapshot.block_tokens = 16;
  snapshot.sequence_active.fill(0);
  snapshot.sequence_length_values.fill(0);
  snapshot.sequence_kv_block_count.fill(0);
  for (auto & row : snapshot.sequence_kv_blocks) {
    row.fill(0);
  }
  snapshot.sequence_recurrent_slot.fill(0);
}

template <class value_type>
inline value_type & bind_or_sink(value_type * ptr, value_type & sink) noexcept {
  const std::array<value_type *, 2> choices{&sink, ptr};
  return *choices[static_cast<size_t>(ptr != nullptr)];
}

}  // namespace emel::memory::detail
