#pragma once

#include <cstdint>

namespace emel::memory::view {

struct any {
  using is_sequence_active_fn = bool (*)(const void * self, int32_t seq_id);
  using sequence_length_fn = int32_t (*)(const void * self, int32_t seq_id);
  using lookup_kv_block_fn = int32_t (*)(const void * self, int32_t seq_id, int32_t pos);
  using lookup_recurrent_slot_fn = int32_t (*)(const void * self, int32_t seq_id);

  const void * self = nullptr;
  is_sequence_active_fn is_sequence_active_impl = nullptr;
  sequence_length_fn sequence_length_impl = nullptr;
  lookup_kv_block_fn lookup_kv_block_impl = nullptr;
  lookup_recurrent_slot_fn lookup_recurrent_slot_impl = nullptr;

  bool is_sequence_active(const int32_t seq_id) const noexcept {
    if (is_sequence_active_impl == nullptr) {
      return false;
    }
    return is_sequence_active_impl(self, seq_id);
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    if (sequence_length_impl == nullptr) {
      return 0;
    }
    return sequence_length_impl(self, seq_id);
  }

  int32_t lookup_kv_block(const int32_t seq_id, const int32_t pos) const noexcept {
    if (lookup_kv_block_impl == nullptr) {
      return -1;
    }
    return lookup_kv_block_impl(self, seq_id, pos);
  }

  int32_t lookup_recurrent_slot(const int32_t seq_id) const noexcept {
    if (lookup_recurrent_slot_impl == nullptr) {
      return -1;
    }
    return lookup_recurrent_slot_impl(self, seq_id);
  }
};

}  // namespace emel::memory::view
