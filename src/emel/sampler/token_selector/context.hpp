#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/token_selector/events.hpp"

namespace emel::sampler::token_selector::action {

struct context {
  const event::select_token * request = nullptr;
  int32_t candidate_count = 0;
  event::selection_policy policy = event::selection_policy::argmax;
  float random_01 = 0.0f;
  int32_t selected_token = -1;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::sampler::token_selector::action
