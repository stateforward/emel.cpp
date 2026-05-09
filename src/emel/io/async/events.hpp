#pragma once

#include <cstdint>
#include <limits>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/async/errors.hpp"

namespace emel::io::async::events {

struct load_window_done;
struct load_window_error;
struct load_window_progress_done;

} // namespace emel::io::async::events

namespace emel::io::async::event {

// Caller-owned window contract. When later phases introduce suspension, this
// storage must outlive every cooperative progress tick for the request.
struct load_window_storage {
  uint64_t file_offset = 0u;
  uint64_t logical_byte_length = 0u;
  uint64_t progress_chunk_bytes = 0u;
  uint64_t scheduler_resource_bytes = std::numeric_limits<uint64_t>::max();
  const void *source_span = nullptr;
  uint64_t source_span_bytes = 0u;
  void *target_buffer = nullptr;
  uint64_t target_window_bytes = 0u;
};

// Caller-owned progress state. It is intentionally separate from actor context
// so the async actor never mirrors dispatch-local request data for resumption.
struct load_window_progress {
  uint64_t bytes_committed = 0u;
  bool cancel_requested = false;
};

struct load_window_request {
  load_window_storage &storage;
  load_window_progress &progress;

  load_window_request(load_window_storage &storage_in,
                      load_window_progress &progress_in) noexcept
      : storage(storage_in), progress(progress_in) {}
};

struct load_window {
  const load_window_request &request;
  emel::callback<void(const events::load_window_progress_done &)> on_progress =
      {};
  emel::callback<void(const events::load_window_done &)> on_done = {};
  emel::callback<void(const events::load_window_error &)> on_error = {};

  explicit load_window(const load_window_request &request_in) noexcept
      : request(request_in) {}
};

} // namespace emel::io::async::event

namespace emel::io::async::events {

struct load_window_done {
  const event::load_window &intent;
  void *target_buffer = nullptr;
  uint64_t bytes_committed = 0u;
};

struct load_window_progress_done {
  const event::load_window &intent;
  void *target_buffer = nullptr;
  uint64_t bytes_committed = 0u;
  uint64_t bytes_delta = 0u;
};

struct load_window_error {
  const event::load_window &intent;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::io::async::events
