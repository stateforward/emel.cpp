#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/kv/sm.hpp"
#include "emel/memory/recurrent/sm.hpp"

namespace emel::memory::hybrid::action {

struct context {
  emel::memory::kv::sm kv = {};
  emel::memory::recurrent::sm recurrent = {};

  int32_t phase_error = EMEL_OK;
  bool phase_out_of_memory = false;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::memory::hybrid::action
