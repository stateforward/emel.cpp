#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/candidate_builder/events.hpp"

namespace emel::sampler::candidate_builder::action {

struct context {
  const event::build * request = nullptr;
  int32_t candidate_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::sampler::candidate_builder::action
