#pragma once

#include <cstdint>

#include "emel/memory/view.hpp"

namespace emel::memory::event {

struct reserve {
  int32_t max_sequences = 0;
  int32_t max_blocks = 0;
  int32_t block_tokens = 16;
  int32_t * error_out = nullptr;
};

struct allocate_sequence {
  int32_t seq_id = 0;
  int32_t * error_out = nullptr;
};

struct allocate_slots {
  int32_t seq_id = 0;
  int32_t token_count = 0;
  int32_t * block_count_out = nullptr;
  int32_t * error_out = nullptr;
};

struct branch_sequence {
  using copy_state_fn =
      bool (*)(int32_t src_slot, int32_t dst_slot, void * user_data, int32_t * error_out);

  int32_t parent_seq_id = 0;
  int32_t child_seq_id = 0;
  copy_state_fn copy_state = nullptr;
  void * copy_state_user_data = nullptr;
  int32_t * error_out = nullptr;
};

struct free_sequence {
  int32_t seq_id = 0;
  int32_t * error_out = nullptr;
};

struct rollback_slots {
  int32_t seq_id = 0;
  int32_t token_count = 0;
  int32_t * block_count_out = nullptr;
  int32_t * error_out = nullptr;
};

struct capture_view {
  emel::memory::view::snapshot * snapshot_out = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::memory::event

namespace emel::memory::events {

struct reserve_done {
  const event::reserve * request = nullptr;
};
struct reserve_error {
  int32_t err = 0;
  const event::reserve * request = nullptr;
};

struct allocate_sequence_done {
  const event::allocate_sequence * request = nullptr;
};
struct allocate_sequence_error {
  int32_t err = 0;
  const event::allocate_sequence * request = nullptr;
};

struct allocate_slots_done {
  int32_t block_count = 0;
  const event::allocate_slots * request = nullptr;
};
struct allocate_slots_error {
  int32_t err = 0;
  const event::allocate_slots * request = nullptr;
};

struct branch_sequence_done {
  const event::branch_sequence * request = nullptr;
};
struct branch_sequence_error {
  int32_t err = 0;
  const event::branch_sequence * request = nullptr;
};

struct free_sequence_done {
  const event::free_sequence * request = nullptr;
};
struct free_sequence_error {
  int32_t err = 0;
  const event::free_sequence * request = nullptr;
};

struct rollback_slots_done {
  int32_t block_count = 0;
  const event::rollback_slots * request = nullptr;
};
struct rollback_slots_error {
  int32_t err = 0;
  const event::rollback_slots * request = nullptr;
};

struct capture_view_done {
  const event::capture_view * request = nullptr;
};
struct capture_view_error {
  int32_t err = 0;
  const event::capture_view * request = nullptr;
};

}  // namespace emel::memory::events
