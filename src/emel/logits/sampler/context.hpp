#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/logits/sampler/events.hpp"

namespace emel::logits::sampler::action {

struct context {
  event::sampler_fn * sampler_fns = nullptr;
  int32_t sampler_count = 0;
};

}  // namespace emel::logits::sampler::action
