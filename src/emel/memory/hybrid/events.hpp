#pragma once

#include <cstdint>

namespace emel::memory::hybrid::event {

struct reserve {
  int32_t kv_size = 0;
  int32_t recurrent_slot_capacity = 0;
  int32_t n_stream = 1;
  int32_t n_pad = 1;
  int32_t n_swa = 0;
  int32_t swa_type = 0;
  const int32_t *seq_to_stream = nullptr;
  int32_t seq_to_stream_count = 0;
  int32_t *error_out = nullptr;
};

struct allocate_sequence {
  int32_t seq_id = 0;
  int32_t slot_count = 0;
  int32_t *error_out = nullptr;
};

struct branch_sequence {
  int32_t seq_id_src = 0;
  int32_t seq_id_dst = 0;
  int32_t *error_out = nullptr;
};

struct free_sequence {
  int32_t seq_id = 0;
  int32_t *error_out = nullptr;
};

} // namespace emel::memory::hybrid::event

namespace emel::memory::hybrid::events {

struct reserve_done {
  const event::reserve *request = nullptr;
};
struct reserve_error {
  int32_t err = 0;
  const event::reserve *request = nullptr;
};

struct allocate_sequence_done {
  const event::allocate_sequence *request = nullptr;
};
struct allocate_sequence_error {
  int32_t err = 0;
  const event::allocate_sequence *request = nullptr;
};

struct branch_sequence_done {
  const event::branch_sequence *request = nullptr;
};
struct branch_sequence_error {
  int32_t err = 0;
  const event::branch_sequence *request = nullptr;
};

struct free_sequence_done {
  const event::free_sequence *request = nullptr;
};
struct free_sequence_error {
  int32_t err = 0;
  const event::free_sequence *request = nullptr;
};

struct hybrid_done {};
struct hybrid_error {
  int32_t err = 0;
};

} // namespace emel::memory::hybrid::events
