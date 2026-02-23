#pragma once

#include <cstdint>

#include "emel/emel.h"

namespace emel::text::jinja::parser::action {

struct context {
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

} // namespace emel::text::jinja::parser::action
