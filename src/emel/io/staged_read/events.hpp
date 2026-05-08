#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/events.hpp"
#include "emel/io/staged_read/errors.hpp"

namespace emel::io::staged_read::events {

struct staged_window_done;
struct staged_window_error;
struct staged_window_batch_done;
struct staged_window_batch_error;

} // namespace emel::io::staged_read::events

namespace emel::io::staged_read::event {

struct staged_window_request {
  uint64_t file_offset = 0u;
  uint64_t logical_byte_length = 0u;
  uint64_t stage_chunk_bytes = 0u;
  const void *source_span = nullptr;
  uint64_t source_span_bytes = 0u;
  void *target_buffer = nullptr;
  uint64_t target_window_bytes = 0u;
};

struct staged_window {
  const staged_window_request &request;
  emel::callback<void(const events::staged_window_done &)> on_done = {};
  emel::callback<void(const events::staged_window_error &)> on_error = {};

  explicit staged_window(const staged_window_request &request_in) noexcept
      : request(request_in) {}
};

struct staged_window_batch {
  std::span<const emel::io::event::tensor_load_span> tensors = {};
  uint64_t stage_chunk_bytes = 0u;
  emel::callback<void(const events::staged_window_batch_done &)> on_done = {};
  emel::callback<void(const events::staged_window_batch_error &)> on_error = {};

  explicit staged_window_batch(
      std::span<const emel::io::event::tensor_load_span> tensors_in,
      const uint64_t stage_chunk_bytes_in) noexcept
      : tensors(tensors_in), stage_chunk_bytes(stage_chunk_bytes_in) {}
};

} // namespace emel::io::staged_read::event

namespace emel::io::staged_read::events {

struct staged_window_done {
  const event::staged_window &intent;
  void *target_buffer = nullptr;
  uint64_t bytes_committed = 0u;
};

struct staged_window_error {
  const event::staged_window &intent;
  emel::error::type err = emel::error::cast(error::none);
};

struct staged_window_batch_done {
  const event::staged_window_batch &intent;
  uint32_t done_count = 0u;
  uint64_t bytes_committed = 0u;
};

struct staged_window_batch_error {
  const event::staged_window_batch &intent;
  emel::error::type err = emel::error::cast(error::none);
  uint32_t failed_index = 0u;
};

} // namespace emel::io::staged_read::events
