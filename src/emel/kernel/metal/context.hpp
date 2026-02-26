#pragma once

#include <cstdint>

namespace emel::kernel::metal::action {

struct context {
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::metal::action
