#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/hybrid/events.hpp"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"

namespace emel::memory::hybrid::action {

inline constexpr int32_t MAX_SEQ = emel::memory::recurrent::action::MAX_SEQ;

struct context {
  emel::memory::kv::sm kv_memory = {};
  emel::memory::recurrent::sm recurrent_memory = {};

  bool reserved = false;

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  event::reserve reserve_request = {};
  event::allocate_sequence allocate_request = {};
  event::branch_sequence branch_request = {};
  event::free_sequence free_request = {};
};

} // namespace emel::memory::hybrid::action
