#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/events.hpp"
#include "emel/io/read/errors.hpp"

namespace emel::io::read::events {

struct read_tensor_done;
struct read_tensor_error;
struct read_tensor_result;
struct read_tensor_batch_done;
struct read_tensor_batch_error;

} // namespace emel::io::read::events

namespace emel::io::read::event {

// Identity of a read/copy request. The caller-provided target buffer span
// (`target_buffer` + `target_buffer_bytes`) remains owned by the caller; the
// read strategy writes copied bytes into it on success and never claims
// residency over the buffer. Filesystem work is outside this actor's RTC
// dispatch: callers provide the already-read immutable source span and any
// external source error through `source_buffer`, `source_buffer_bytes`, and
// `source_error`.
struct read_tensor_request {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  std::string_view file_path = {};
  const void *source_buffer = nullptr;
  uint64_t source_buffer_bytes = 0u;
  emel::error::type source_error = emel::error::cast(error::none);
  void *target_buffer = nullptr;
  uint64_t target_buffer_bytes = 0u;
};

struct read_tensor {
  const read_tensor_request &request;
  // Required for valid read requests so callers learn the copied byte count.
  // Phase 215 consumes `read_tensor_done` at the tensor residency boundary.
  emel::callback<void(const events::read_tensor_done &)> on_done = {};
  emel::callback<void(const events::read_tensor_error &)> on_error = {};

  explicit read_tensor(const read_tensor_request &request_in) noexcept
      : request(request_in) {}
};

struct read_tensor_batch {
  std::span<const emel::io::event::tensor_load_span> tensors = {};
  emel::callback<void(const events::read_tensor_batch_done &)> on_done = {};
  emel::callback<void(const events::read_tensor_batch_error &)> on_error = {};

  explicit read_tensor_batch(
      std::span<const emel::io::event::tensor_load_span> tensors_in) noexcept
      : tensors(tensors_in) {}
};

} // namespace emel::io::read::event

namespace emel::io::read::events {

struct read_tensor_done {
  const event::read_tensor &request;
  // Bytes copied into the caller-owned target buffer from the source span.
  uint64_t bytes_copied = 0u;
  // Echo of the caller-provided target buffer base for downstream tensor
  // bind validation. The read strategy never reallocates the buffer; this
  // pointer is identical to `request.request.target_buffer` on success.
  void *target_buffer = nullptr;
};

struct read_tensor_error {
  const event::read_tensor &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct read_tensor_result {
  bool accepted = false;
  bool ok = false;
  emel::error::type err = emel::error::cast(error::none);
  uint64_t bytes_copied = 0u;
  void *target_buffer = nullptr;
};

struct read_tensor_batch_done {
  const event::read_tensor_batch &request;
  uint32_t done_count = 0u;
  uint64_t bytes_copied = 0u;
};

struct read_tensor_batch_error {
  const event::read_tensor_batch &request;
  emel::error::type err = emel::error::cast(error::none);
  uint32_t failed_index = 0u;
};

} // namespace emel::io::read::events
