#pragma once

#include <cstdint>

namespace emel::kernel::cuda::action {

struct context {
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::cuda::action
