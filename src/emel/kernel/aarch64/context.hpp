#pragma once

#include <cstdint>

namespace emel::kernel::aarch64::action {

struct context {
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::aarch64::action
