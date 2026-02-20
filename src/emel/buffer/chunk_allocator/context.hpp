#pragma once

#include <array>
#include <cstdint>
#include <limits>

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
  int32_t phase_error = EMEL_OK;
  uint32_t step = 0;
};

}  // namespace emel::buffer::chunk_allocator::action
