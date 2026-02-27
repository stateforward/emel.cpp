#pragma once

#include <array>
#include <cstdint>

#include "emel/memory/kv/detail.hpp"

namespace emel::memory::kv::action {

struct context {
  int32_t max_sequences = detail::max_sequences;
  int32_t max_blocks = detail::max_blocks;
  int32_t block_tokens = detail::default_block_tokens;

  detail::block_pool block_refs = {};
  std::array<bool, detail::max_sequences> sequence_active = {};
  std::array<int32_t, detail::max_sequences> sequence_length = {};
  std::array<int32_t, detail::max_sequences> sequence_block_count = {};
  std::array<std::array<uint16_t, detail::max_blocks_per_sequence>, detail::max_sequences>
      seq_to_blocks = {};

  std::array<uint16_t, detail::max_blocks> free_stack = {};
  int32_t free_count = 0;
};

}  // namespace emel::memory::kv::action
