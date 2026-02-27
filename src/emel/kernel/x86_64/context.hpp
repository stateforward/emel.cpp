#pragma once

#include <cstdint>

#include "emel/kernel/x86_64/detail.hpp"

namespace emel::kernel::x86_64::action {

struct context {
  const bool avx2_available = detail::detect_avx2();
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::x86_64::action
