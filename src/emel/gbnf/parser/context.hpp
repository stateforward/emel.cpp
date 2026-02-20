#pragma once

#include "emel/emel.h"
#include <cstdint>

namespace emel::gbnf::parser::action {

struct context {
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

} // namespace emel::gbnf::parser::action
