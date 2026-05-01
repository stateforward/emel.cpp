#pragma once

#include <cstdint>

#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/x86_64/sm.hpp"

namespace emel::kernel::action {

struct context {
  x86_64::sm x86_64_actor = {};
  aarch64::sm aarch64_actor = {};
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::action
