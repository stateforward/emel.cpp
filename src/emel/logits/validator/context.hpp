#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/logits/validator/events.hpp"

namespace emel::logits::validator::action {

struct context {
  const event::build * request = nullptr;
  int32_t candidate_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::logits::validator::action
