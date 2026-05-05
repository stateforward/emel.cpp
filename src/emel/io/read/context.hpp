#pragma once

#include "emel/io/read/errors.hpp"

namespace emel::io::read::action {

// The read strategy is allocation-free at the boundary and never claims tensor
// residency. Phase 212 keeps `context` empty/persistent-only: per AGENTS.md
// context rules, dispatch-local request data (request payload, target-buffer
// pointer, offset, length, error/status codes, output pointers, phase
// indices) lives only in typed internal events. Future phases (213/214) MAY
// add persistent actor-owned tuning state here (for example a configurable
// I/O block size budget), but only when that state survives across top-level
// dispatch calls and never mirrors per-invocation request fields.
struct context {};

} // namespace emel::io::read::action
