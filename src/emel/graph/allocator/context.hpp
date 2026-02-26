#pragma once

#include <cstdint>

namespace emel::graph::allocator::action {

struct context {
  uint32_t dispatch_generation = 0;
};

}  // namespace emel::graph::allocator::action
