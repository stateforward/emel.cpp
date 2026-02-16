#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer/chunk_allocator/events.hpp"
#include "emel/emel.h"

namespace emel::buffer::chunk_allocator::action {

inline constexpr int32_t k_max_chunks = 16;
inline constexpr int32_t k_max_free_blocks = 256;
inline constexpr uint64_t k_default_alignment = 16;
inline constexpr uint64_t k_max_chunk_limit = std::numeric_limits<uint64_t>::max() / 2;

struct free_block {
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct chunk_data {
  std::array<free_block, k_max_free_blocks> free_blocks = {};
  int32_t free_block_count = 0;
  uint64_t max_size = 0;
};

struct context {
  uint64_t alignment = k_default_alignment;
  uint64_t max_chunk_size = k_max_chunk_limit;
  std::array<chunk_data, k_max_chunks> chunks = {};
  int32_t chunk_count = 0;

  uint64_t pending_alignment = k_default_alignment;
  uint64_t pending_max_chunk_size = k_max_chunk_limit;

  uint64_t request_size = 0;
  uint64_t request_alignment = k_default_alignment;
  uint64_t request_max_chunk_size = k_max_chunk_limit;
  uint64_t aligned_request_size = 0;
  int32_t request_chunk = -1;
  uint64_t request_offset = 0;

  int32_t selected_chunk = -1;
  int32_t selected_block = -1;

  int32_t result_chunk = -1;
  uint64_t result_offset = 0;
  uint64_t result_size = 0;
  uint32_t step = 0;
};

namespace detail {

inline int32_t normalize_error(const int32_t err, const int32_t fallback) noexcept {
  if (err != 0) return err;
  if (fallback != 0) return fallback;
  return EMEL_ERR_BACKEND;
}

inline uint64_t clamp_chunk_size_limit(const uint64_t value) noexcept {
  return value > k_max_chunk_limit ? k_max_chunk_limit : value;
}

inline bool valid_alignment(const uint64_t alignment) noexcept { return alignment > 0; }

inline bool add_overflow(const uint64_t lhs, const uint64_t rhs, uint64_t & out) noexcept {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return true;
  }
  out = lhs + rhs;
  return false;
}

inline bool align_up(const uint64_t value, const uint64_t alignment, uint64_t & out) noexcept {
  if (!valid_alignment(alignment)) {
    return false;
  }
  const uint64_t rem = value % alignment;
  if (rem == 0) {
    out = value;
    return true;
  }
  const uint64_t inc = alignment - rem;
  return !add_overflow(value, inc, out);
}

inline void reset_chunks(context & c) noexcept {
  c.chunks = {};
  c.chunk_count = 0;
}

inline void remove_block(chunk_data & chunk, const int32_t idx) noexcept {
  if (idx < 0 || idx >= chunk.free_block_count) {
    return;
  }
  for (int32_t i = idx; i + 1 < chunk.free_block_count; ++i) {
    chunk.free_blocks[i] = chunk.free_blocks[i + 1];
  }
  chunk.free_block_count -= 1;
}

inline bool insert_block(chunk_data & chunk, const uint64_t offset, const uint64_t size) noexcept {
  if (size == 0) {
    return true;
  }
  if (chunk.free_block_count >= k_max_free_blocks) {
    return false;
  }

  int32_t pos = 0;
  while (pos < chunk.free_block_count && chunk.free_blocks[pos].offset < offset) {
    pos += 1;
  }
  for (int32_t i = chunk.free_block_count; i > pos; --i) {
    chunk.free_blocks[i] = chunk.free_blocks[i - 1];
  }
  chunk.free_blocks[pos] = free_block{.offset = offset, .size = size};
  chunk.free_block_count += 1;
  return true;
}

inline int64_t reuse_factor(
    const chunk_data & chunk, const free_block & block, const uint64_t alloc_size) noexcept {
  uint64_t end = 0;
  if (add_overflow(block.offset, alloc_size, end)) {
    return std::numeric_limits<int64_t>::min();
  }
  if (chunk.max_size >= end) {
    const uint64_t slack = chunk.max_size - end;
    if (slack > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(slack);
  }
  const uint64_t deficit = end - chunk.max_size;
  if (deficit > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::numeric_limits<int64_t>::min();
  }
  return -static_cast<int64_t>(deficit);
}

inline void select_best_non_last(context & c) noexcept {
  c.selected_chunk = -1;
  c.selected_block = -1;

  uint64_t best_size = std::numeric_limits<uint64_t>::max();
  for (int32_t chunk_idx = 0; chunk_idx < c.chunk_count; ++chunk_idx) {
    const auto & chunk = c.chunks[chunk_idx];
    for (int32_t i = 0; i < chunk.free_block_count - 1; ++i) {
      const auto & block = chunk.free_blocks[i];
      if (block.size >= c.aligned_request_size && block.size <= best_size) {
        c.selected_chunk = chunk_idx;
        c.selected_block = i;
        best_size = block.size;
      }
    }
  }
}

inline void select_best_last(context & c) noexcept {
  int64_t best_reuse = std::numeric_limits<int64_t>::min();
  for (int32_t chunk_idx = 0; chunk_idx < c.chunk_count; ++chunk_idx) {
    const auto & chunk = c.chunks[chunk_idx];
    if (chunk.free_block_count <= 0) {
      continue;
    }
    const int32_t last_idx = chunk.free_block_count - 1;
    const auto & block = chunk.free_blocks[last_idx];
    if (block.size < c.aligned_request_size) {
      continue;
    }
    const int64_t cur_reuse = reuse_factor(chunk, block, c.aligned_request_size);
    const bool better_reuse = best_reuse < 0 && cur_reuse > best_reuse;
    const bool better_fit = cur_reuse >= 0 && cur_reuse < best_reuse;
    if (c.selected_chunk < 0 || better_reuse || better_fit) {
      c.selected_chunk = chunk_idx;
      c.selected_block = last_idx;
      best_reuse = cur_reuse;
    }
  }
}

inline bool create_chunk(context & c, const uint64_t min_size, int32_t & out_chunk) noexcept {
  if (c.chunk_count >= k_max_chunks) {
    return false;
  }

  uint64_t chunk_size =
    min_size > c.request_max_chunk_size ? min_size : c.request_max_chunk_size;
  if (c.chunk_count == k_max_chunks - 1) {
    chunk_size = k_max_chunk_limit;
  }

  auto & chunk = c.chunks[c.chunk_count];
  chunk = {};
  chunk.free_block_count = 1;
  chunk.free_blocks[0] = free_block{
    .offset = 0,
    .size = chunk_size,
  };
  out_chunk = c.chunk_count;
  c.chunk_count += 1;
  return true;
}

inline bool allocate_from_selected(context & c) noexcept {
  if (c.selected_chunk < 0 || c.selected_chunk >= c.chunk_count) {
    return false;
  }
  auto & chunk = c.chunks[c.selected_chunk];
  if (c.selected_block < 0 || c.selected_block >= chunk.free_block_count) {
    return false;
  }

  auto & block = chunk.free_blocks[c.selected_block];
  if (block.size < c.aligned_request_size) {
    return false;
  }

  c.result_chunk = c.selected_chunk;
  c.result_offset = block.offset;
  c.result_size = c.aligned_request_size;

  block.offset += c.aligned_request_size;
  block.size -= c.aligned_request_size;
  if (block.size == 0) {
    remove_block(chunk, c.selected_block);
  }

  uint64_t end = 0;
  if (add_overflow(c.result_offset, c.aligned_request_size, end)) {
    return false;
  }
  if (end > chunk.max_size) {
    chunk.max_size = end;
  }
  return true;
}

inline bool free_bytes(context & c) noexcept {
  if (c.request_chunk < 0 || c.request_chunk >= c.chunk_count) {
    return false;
  }

  auto & chunk = c.chunks[c.request_chunk];
  uint64_t release_end = 0;
  if (add_overflow(c.request_offset, c.aligned_request_size, release_end)) {
    return false;
  }

  for (int32_t i = 0; i < chunk.free_block_count; ++i) {
    auto & block = chunk.free_blocks[i];

    uint64_t block_end = 0;
    if (!add_overflow(block.offset, block.size, block_end) && block_end == c.request_offset) {
      block.size += c.aligned_request_size;
      if (i < chunk.free_block_count - 1) {
        auto & next = chunk.free_blocks[i + 1];
        uint64_t merged_end = 0;
        if (!add_overflow(block.offset, block.size, merged_end) && merged_end == next.offset) {
          block.size += next.size;
          remove_block(chunk, i + 1);
        }
      }
      return true;
    }

    if (release_end == block.offset) {
      block.offset = c.request_offset;
      block.size += c.aligned_request_size;
      if (i > 0) {
        auto & prev = chunk.free_blocks[i - 1];
        uint64_t prev_end = 0;
        if (!add_overflow(prev.offset, prev.size, prev_end) && prev_end == block.offset) {
          prev.size += block.size;
          remove_block(chunk, i);
        }
      }
      return true;
    }
  }

  return insert_block(chunk, c.request_offset, c.aligned_request_size);
}

}  // namespace detail

struct begin_configure {
  void operator()(const event::configure & ev, context & c) const noexcept {
    c.pending_alignment = ev.alignment;
    c.pending_max_chunk_size = ev.max_chunk_size;
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_validate_configure {
  void operator()(const event::validate_configure & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (!detail::valid_alignment(c.pending_alignment) || c.pending_max_chunk_size == 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct run_apply_configure {
  void operator()(const event::apply_configure & ev, context & c) const noexcept {
    c.alignment = c.pending_alignment;
    c.max_chunk_size = detail::clamp_chunk_size_limit(c.pending_max_chunk_size);
    detail::reset_chunks(c);
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct begin_allocate {
  void operator()(const event::allocate & ev, context & c) const noexcept {
    c.request_size = ev.size;
    c.request_alignment = ev.alignment != 0 ? ev.alignment : c.alignment;
    c.request_max_chunk_size = ev.max_chunk_size != 0
      ? detail::clamp_chunk_size_limit(ev.max_chunk_size)
      : c.max_chunk_size;
    c.result_chunk = -1;
    c.result_offset = 0;
    c.result_size = 0;
    c.selected_chunk = -1;
    c.selected_block = -1;
    if (ev.chunk_out != nullptr) {
      *ev.chunk_out = -1;
    }
    if (ev.offset_out != nullptr) {
      *ev.offset_out = 0;
    }
    if (ev.aligned_size_out != nullptr) {
      *ev.aligned_size_out = 0;
    }
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_validate_allocate {
  void operator()(const event::validate_allocate & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (c.request_size == 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (!detail::valid_alignment(c.request_alignment) || c.request_max_chunk_size == 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (!detail::align_up(c.request_size, c.request_alignment, c.aligned_request_size)) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct run_select_block {
  void operator()(const event::select_block & ev, context & c) const noexcept {
    c.selected_chunk = -1;
    c.selected_block = -1;
    detail::select_best_non_last(c);
    if (c.selected_chunk < 0) {
      detail::select_best_last(c);
    }
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_ensure_chunk {
  void operator()(const event::ensure_chunk & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (c.selected_chunk < 0) {
      int32_t new_chunk = -1;
      if (!detail::create_chunk(c, c.aligned_request_size, new_chunk)) {
        err = EMEL_ERR_BACKEND;
      } else {
        c.selected_chunk = new_chunk;
        c.selected_block = 0;
      }
    }
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct run_commit_allocate {
  void operator()(const event::commit_allocate & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (!detail::allocate_from_selected(c)) {
      err = EMEL_ERR_BACKEND;
    }
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct begin_release {
  void operator()(const event::release & ev, context & c) const noexcept {
    c.request_chunk = ev.chunk;
    c.request_offset = ev.offset;
    c.request_size = ev.size;
    c.request_alignment = ev.alignment != 0 ? ev.alignment : c.alignment;
    c.aligned_request_size = 0;
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_validate_release {
  void operator()(const event::validate_release & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (c.request_chunk < 0 || c.request_chunk >= c.chunk_count || c.request_size == 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (!detail::align_up(c.request_size, c.request_alignment, c.aligned_request_size)) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else {
      uint64_t end = 0;
      if (detail::add_overflow(c.request_offset, c.aligned_request_size, end) ||
          end > c.chunks[c.request_chunk].max_size) {
        err = EMEL_ERR_INVALID_ARGUMENT;
      }
    }
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct run_merge_release {
  void operator()(const event::merge_release & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (!detail::free_bytes(c)) {
      err = EMEL_ERR_BACKEND;
    }
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct begin_reset {
  void operator()(const event::reset & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_apply_reset {
  void operator()(const event::apply_reset & ev, context & c) const noexcept {
    detail::reset_chunks(c);
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_configure_done {
  void operator()(const events::configure_done & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_configure_error {
  void operator()(const events::configure_error & ev, context & c) const noexcept {
    const int32_t err = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct on_allocate_done {
  void operator()(const events::allocate_done & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_allocate_error {
  void operator()(const events::allocate_error & ev, context & c) const noexcept {
    const int32_t err = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct on_release_done {
  void operator()(const events::release_done & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_release_error {
  void operator()(const events::release_error & ev, context & c) const noexcept {
    const int32_t err = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct on_reset_done {
  void operator()(const events::reset_done & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_reset_error {
  void operator()(const events::reset_error & ev, context & c) const noexcept {
    const int32_t err = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    if (ev.error_out != nullptr) *ev.error_out = err;
    c.step += 1;
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & c) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
    }
    c.step += 1;
  }
};

inline constexpr begin_configure begin_configure{};
inline constexpr run_validate_configure run_validate_configure{};
inline constexpr run_apply_configure run_apply_configure{};
inline constexpr begin_allocate begin_allocate{};
inline constexpr run_validate_allocate run_validate_allocate{};
inline constexpr run_select_block run_select_block{};
inline constexpr run_ensure_chunk run_ensure_chunk{};
inline constexpr run_commit_allocate run_commit_allocate{};
inline constexpr begin_release begin_release{};
inline constexpr run_validate_release run_validate_release{};
inline constexpr run_merge_release run_merge_release{};
inline constexpr begin_reset begin_reset{};
inline constexpr run_apply_reset run_apply_reset{};
inline constexpr on_configure_done on_configure_done{};
inline constexpr on_configure_error on_configure_error{};
inline constexpr on_allocate_done on_allocate_done{};
inline constexpr on_allocate_error on_allocate_error{};
inline constexpr on_release_done on_release_done{};
inline constexpr on_release_error on_release_error{};
inline constexpr on_reset_done on_reset_done{};
inline constexpr on_reset_error on_reset_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::buffer::chunk_allocator::action
