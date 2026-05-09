#pragma once

#include "emel/error/error.hpp"
#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"

namespace emel::io::loader::detail {

struct runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type strategy_err = emel::error::cast(error::none);
  bool accepted = false;
  bool ok = false;
  bool partial = false;
  bool done = false;
  uint64_t bytes_copied = 0u;
  uint64_t bytes_delta = 0u;
  void *buffer = nullptr;
};

struct batch_runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type strategy_err = emel::error::cast(error::none);
  bool accepted = false;
  bool ok = false;
  bool partial = false;
  bool done = false;
  uint32_t done_count = 0u;
  uint64_t bytes_done = 0u;
  uint64_t bytes_delta = 0u;
  uint32_t failed_index = 0u;
  uint32_t current_index = 0u;
  uint32_t done_delta = 0u;
  uint64_t bytes_before = 0u;
};

struct load_tensor_runtime {
  const event::load_tensor &request;
  runtime_status &ctx;
};

struct load_tensor_batch_runtime {
  const event::load_tensor_batch &request;
  batch_runtime_status &status;
};

inline uint64_t compute_staged_chunk_bytes(const uint64_t requested,
                                           const uint64_t logical) noexcept {
  const uint64_t requested_is_smaller =
      static_cast<uint64_t>(requested < logical);
  return (requested * requested_is_smaller) +
         (logical * (1u - requested_is_smaller));
}

inline uint64_t compute_async_chunk_bytes(const uint64_t requested,
                                          const uint64_t logical) noexcept {
  const uint64_t requested_nonzero = static_cast<uint64_t>(requested > 0u);
  const uint64_t bounded_requested =
      (requested * requested_nonzero) +
      (event::k_default_staged_read_chunk_bytes * (1u - requested_nonzero));
  return compute_staged_chunk_bytes(bounded_requested, logical);
}

} // namespace emel::io::loader::detail
