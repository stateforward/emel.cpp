#pragma once

#include <cstdint>

#include "emel/emel.h"

namespace emel::generator::action {

inline constexpr int32_t MAX_GENERATION_STEPS = 4096;

struct context {
  int32_t tokens_generated = 0;
  int32_t max_tokens = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::generator::action
