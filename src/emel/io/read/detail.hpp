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
  uint64_t bytes_copied = 0u;
};

struct read_tensor_batch_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  uint32_t done_count = 0u;
  uint64_t bytes_copied = 0u;
  uint32_t failed_index = 0u;
};

inline void ignore_read_tensor_done(void *,
                                    const events::read_tensor_done &) noexcept {
}

inline void
ignore_read_tensor_error(void *, const events::read_tensor_error &) noexcept {}

// Internal-only carrier that bridges a public `event::read_tensor` trigger
// to internal completion progress without copying request payload into
// `action::context`. Per AGENTS.md, internal events MAY use mutable
// reference fields when they are not publicly exposed.
struct read_tensor_runtime {
  const event::read_tensor &request;
  read_attempt_status &status;
};

struct read_tensor_batch_runtime {
  const event::read_tensor_batch &request;
  read_tensor_batch_status &status;
};

} // namespace emel::io::read::detail
