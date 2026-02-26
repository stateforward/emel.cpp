#pragma once

#include <cstdint>

namespace emel::graph::processor::action {

struct context {
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::graph::processor::action
