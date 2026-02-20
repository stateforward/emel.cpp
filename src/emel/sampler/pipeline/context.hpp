#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/pipeline/events.hpp"

namespace emel::sampler::pipeline::action {

struct context {
  const event::sample * request = nullptr;
  int32_t candidate_count = 0;
  int32_t sampler_count = 0;
  int32_t sampler_index = 0;
  int32_t selected_token = -1;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::sampler::pipeline::action
