#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/memory/events.hpp"
#include "emel/memory/recurrent/detail.hpp"
#include "emel/memory/recurrent/errors.hpp"
#include "emel/memory/view.hpp"

namespace emel::memory::recurrent::event {

using reserve = emel::memory::event::reserve;
using allocate_sequence = emel::memory::event::allocate_sequence;
using allocate_slots = emel::memory::event::allocate_slots;
using branch_sequence = emel::memory::event::branch_sequence;
using free_sequence = emel::memory::event::free_sequence;
using rollback_slots = emel::memory::event::rollback_slots;
using capture_view = emel::memory::event::capture_view;

struct reserve_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
  int32_t resolved_max_sequences = 0;
  int32_t resolved_slots = 0;
};

struct allocate_sequence_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
  int32_t slot_id = recurrent::detail::invalid_slot;
  bool slot_activated = false;
};

struct allocate_slots_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
  int32_t block_count = 0;
  int32_t old_length = 0;
  int32_t new_length = 0;
};

struct branch_sequence_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
  int32_t child_slot = recurrent::detail::invalid_slot;
  bool slot_activated = false;
  bool copy_accepted = false;
  int32_t copy_error = static_cast<int32_t>(emel::error::cast(error::none));
};

struct free_sequence_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
  int32_t slot_id = recurrent::detail::invalid_slot;
  bool slot_deactivated = false;
};

struct rollback_slots_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
  int32_t block_count = 0;
  int32_t current_length = 0;
  int32_t new_length = 0;
};

struct capture_view_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool accepted = false;
  emel::error::type operation_error = emel::error::cast(error::none);
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
  emel::memory::view::snapshot & snapshot_out;
  int32_t & error_code_out;
  bool has_snapshot_out = false;
};

}  // namespace emel::memory::recurrent::event

namespace emel::memory::recurrent {

namespace events = emel::memory::events;

}  // namespace emel::memory::recurrent
