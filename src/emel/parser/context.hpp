#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/parser/events.hpp"

namespace emel::parser::action {

struct context {
  event::parse_model request = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::parser::action
