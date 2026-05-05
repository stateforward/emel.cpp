#pragma once

#include "emel/error/error.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/events.hpp"

namespace emel::io::read::detail {

// Per-attempt status carrier observed by guards and mutated by entry actions
// while a single dispatch progresses through the boundary transition chain.
struct read_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
};

// Internal-only carrier that bridges a public `event::read_tensor` trigger
// to internal completion progress without copying request payload into
// `action::context`. Per AGENTS.md, internal events MAY use mutable
// reference fields when they are not publicly exposed.
struct read_tensor_runtime {
  const event::read_tensor &request;
  read_attempt_status &status;
};

} // namespace emel::io::read::detail
