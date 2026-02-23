#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/recurrent/events.hpp"

namespace emel::memory::recurrent::action {

inline constexpr int32_t MAX_SEQ = 256;
inline constexpr int32_t SLOT_NONE = -1;

struct context {
  int32_t slot_capacity = 0;
  int32_t active_count = 0;
  std::array<int32_t, MAX_SEQ> seq_to_slot = {};
  std::array<uint8_t, MAX_SEQ> slot_active = {};

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  event::reserve reserve_request = {};
  event::allocate_sequence allocate_request = {};
  event::branch_sequence branch_request = {};
  event::free_sequence free_request = {};

  context();
};

} // namespace emel::memory::recurrent::action
