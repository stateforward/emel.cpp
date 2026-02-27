#pragma once

#include <cstdint>

#include "emel/kernel/aarch64/detail.hpp"

namespace emel::kernel::aarch64::action {

struct context {
  const bool neon_available = detail::detect_neon();
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::aarch64::action
