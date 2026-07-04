#pragma once

#include "emel/io/staged_read/errors.hpp"

namespace emel::io::staged_read::action {

// Persistent actor state: the platform staged-copy capability, injected at
// construction. The production default is the compile-time platform probe;
// ports without the staged path and tests exercising the modeled
// platform-unsupported routes construct the actor with false. Guards read it
// as a pure predicate; no dispatch-local payload lives here.

struct context {
  bool platform_supported = EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED != 0;
};

} // namespace emel::io::staged_read::action
