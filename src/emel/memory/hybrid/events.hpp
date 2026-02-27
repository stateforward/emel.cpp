#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/memory/events.hpp"
#include "emel/memory/hybrid/errors.hpp"
#include "emel/memory/view.hpp"

namespace emel::memory::hybrid::event {

using reserve = emel::memory::event::reserve;
using allocate_sequence = emel::memory::event::allocate_sequence;
using allocate_slots = emel::memory::event::allocate_slots;
using branch_sequence = emel::memory::event::branch_sequence;
using free_sequence = emel::memory::event::free_sequence;
using rollback_slots = emel::memory::event::rollback_slots;
using capture_view = emel::memory::event::capture_view;

struct reserve_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
};

struct allocate_sequence_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  bool rollback_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
  int32_t rollback_error = 0;
};

struct allocate_slots_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  bool rollback_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
  int32_t rollback_error = 0;
  int32_t kv_block_count = 0;
};

struct branch_sequence_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  bool rollback_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
  int32_t rollback_error = 0;
};

struct free_sequence_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
};

struct rollback_slots_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
  int32_t kv_block_count = 0;
};

struct capture_view_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool kv_accepted = false;
  bool recurrent_accepted = false;
  int32_t kv_error = 0;
  int32_t recurrent_error = 0;
};

struct reserve_runtime {
  const reserve & request;
  reserve_ctx & ctx;
  int32_t & error_code_out;
};

struct allocate_sequence_runtime {
  const allocate_sequence & request;
  allocate_sequence_ctx & ctx;
  int32_t & error_code_out;
};

struct allocate_slots_runtime {
  const allocate_slots & request;
  allocate_slots_ctx & ctx;
  int32_t & block_count_out;
  int32_t & error_code_out;
};

struct branch_sequence_runtime {
  const branch_sequence & request;
  branch_sequence_ctx & ctx;
  int32_t & error_code_out;
};

struct free_sequence_runtime {
  const free_sequence & request;
  free_sequence_ctx & ctx;
  int32_t & error_code_out;
};

struct rollback_slots_runtime {
  const rollback_slots & request;
  rollback_slots_ctx & ctx;
  int32_t & block_count_out;
  int32_t & error_code_out;
};

struct capture_view_runtime {
  const capture_view & request;
  capture_view_ctx & ctx;
  emel::memory::view::snapshot & kv_snapshot;
  emel::memory::view::snapshot & recurrent_snapshot;
  emel::memory::view::snapshot & snapshot_out;
  int32_t & error_code_out;
  bool has_snapshot_out = false;
};

}  // namespace emel::memory::hybrid::event

namespace emel::memory::hybrid {

namespace events = emel::memory::events;

}  // namespace emel::memory::hybrid
