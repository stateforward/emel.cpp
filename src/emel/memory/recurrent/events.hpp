#pragma once

#include <cstdint>

namespace emel::memory::recurrent::event {

struct reserve {
  int32_t slot_capacity = 0;
  int32_t *error_out = nullptr;
};

struct allocate_sequence {
  int32_t seq_id = 0;
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

} // namespace emel::memory::recurrent::event

namespace emel::memory::recurrent::events {

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

struct recurrent_done {};
struct recurrent_error {
  int32_t err = 0;
};

} // namespace emel::memory::recurrent::events
