#pragma once

#include <cstdint>

namespace emel::kernel::aarch64::action {

struct context {
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::aarch64::action
