#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace emel::memory::view {

inline constexpr int32_t MAX_SEQUENCES = 256;
inline constexpr int32_t MAX_BLOCKS_PER_SEQUENCE = 4096;
inline constexpr uint16_t INVALID_KV_BLOCK = UINT16_MAX;

struct snapshot {
  int32_t max_sequences = 0;
  int32_t block_tokens = 16;

  std::array<uint8_t, MAX_SEQUENCES> sequence_active = {};
  std::array<int32_t, MAX_SEQUENCES> sequence_length_values = {};
  std::array<int32_t, MAX_SEQUENCES> sequence_kv_block_count = {};
  std::array<std::array<uint16_t, MAX_BLOCKS_PER_SEQUENCE>, MAX_SEQUENCES> sequence_kv_blocks = {};
  std::array<int32_t, MAX_SEQUENCES> sequence_recurrent_slot = {};

  bool valid_seq_id(const int32_t seq_id) const noexcept {
    return seq_id >= 0 && seq_id < max_sequences && seq_id < MAX_SEQUENCES;
  }

  bool is_sequence_active(const int32_t seq_id) const noexcept {
    if (!valid_seq_id(seq_id)) {
      return false;
    }
    return sequence_active[static_cast<size_t>(seq_id)] != 0;
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    if (!is_sequence_active(seq_id)) {
      return 0;
    }
    return sequence_length_values[static_cast<size_t>(seq_id)];
  }

  int32_t lookup_kv_block(const int32_t seq_id, const int32_t pos) const noexcept {
    if (!is_sequence_active(seq_id) || pos < 0 || block_tokens <= 0) {
      return -1;
    }
    const int32_t length = sequence_length_values[static_cast<size_t>(seq_id)];
    if (pos >= length) {
      return -1;
    }
    const int32_t block_count = sequence_kv_block_count[static_cast<size_t>(seq_id)];
    if (block_count <= 0 || block_count > MAX_BLOCKS_PER_SEQUENCE) {
      return -1;
    }
    const int32_t logical_block = pos / block_tokens;
    if (logical_block < 0 || logical_block >= block_count) {
      return -1;
    }
    const uint16_t block_id = sequence_kv_blocks[static_cast<size_t>(seq_id)]
                              [static_cast<size_t>(logical_block)];
    if (block_id == INVALID_KV_BLOCK) {
      return -1;
    }
    return static_cast<int32_t>(block_id);
  }

  int32_t lookup_recurrent_slot(const int32_t seq_id) const noexcept {
    if (!is_sequence_active(seq_id)) {
      return -1;
    }
    return sequence_recurrent_slot[static_cast<size_t>(seq_id)];
  }
};

}  // namespace emel::memory::view
