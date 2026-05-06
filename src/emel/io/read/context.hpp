#pragma once

#include "emel/io/read/errors.hpp"

namespace emel::io::read::action {

// The read strategy is allocation-free at the boundary and never claims tensor
// residency. `context` is intentionally empty because dispatch-local request
// data (request payload, target-buffer pointer, source span, offset, length,
// error/status codes, output pointers, and phase indices) lives only in typed
// internal events.
struct context {};

} // namespace emel::io::read::action
