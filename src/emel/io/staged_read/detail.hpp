#pragma once

#include "emel/error/error.hpp"
#include "emel/io/staged_read/errors.hpp"
#include "emel/io/staged_read/events.hpp"

namespace emel::io::staged_read::detail {

struct staged_window_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
};

// INTERNAL-only synchronous carrier; mutable status is not exposed publicly.
struct staged_window_runtime {
  const event::staged_window &request;
  staged_window_attempt_status &status;
};

struct staged_window_batch_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  uint32_t done_count = 0u;
  uint64_t bytes_committed = 0u;
  uint32_t failed_index = 0u;
};

struct staged_window_batch_runtime {
  const event::staged_window_batch &request;
  staged_window_batch_attempt_status &status;
};

inline uint64_t compute_stage_chunk_bytes(const uint64_t requested,
                                          const uint64_t logical) noexcept {
  const uint64_t requested_is_smaller =
      static_cast<uint64_t>(requested < logical);
  return (requested * requested_is_smaller) +
         (logical * (1u - requested_is_smaller));
}

} // namespace emel::io::staged_read::detail
