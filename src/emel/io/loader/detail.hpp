#pragma once

#include "emel/error/error.hpp"
#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"

namespace emel::io::loader::detail {

struct runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type strategy_err = emel::error::cast(error::none);
  bool ok = false;
  uint64_t bytes_copied = 0u;
  void *buffer = nullptr;
};

struct batch_runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type strategy_err = emel::error::cast(error::none);
  bool accepted = false;
  bool ok = false;
  uint32_t done_count = 0u;
  uint64_t bytes_done = 0u;
  uint32_t failed_index = 0u;
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

} // namespace emel::io::loader::detail
