#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/read/errors.hpp"

namespace emel::io::read::events {

struct read_tensor_done;
struct read_tensor_error;

} // namespace emel::io::read::events

namespace emel::io::read::event {

// Identity of a read/copy request. The caller-provided target buffer span
// (`target_buffer` + `target_buffer_bytes`) remains owned by the caller; the
// read strategy writes copied bytes into it on success and never claims
// residency over the buffer. The action layer copies `file_path` into a
// bounded stack buffer before any platform call (Phase 214); embedded NUL
// bytes are rejected by the file-path guard introduced in Phase 213.
struct read_tensor_request {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  std::string_view file_path = {};
  void *target_buffer = nullptr;
  uint64_t target_buffer_bytes = 0u;
};

struct read_tensor {
  const read_tensor_request &request;
  // Required for valid read requests so callers learn the actual copied byte
  // count; optional for the boundary phase since accepted requests fail
  // closed at `unsupported_platform`. Phase 215 will tighten this contract
  // when tensor-side integration consumes `read_tensor_done`.
  emel::callback<void(const events::read_tensor_done &)> on_done = {};
  emel::callback<void(const events::read_tensor_error &)> on_error = {};

  explicit read_tensor(const read_tensor_request &request_in) noexcept
      : request(request_in) {}
};

} // namespace emel::io::read::event

namespace emel::io::read::events {

struct read_tensor_done {
  const event::read_tensor &request;
  // Bytes actually copied into the caller-owned target buffer. Phase 214 will
  // populate this from the platform read path; the boundary phase always
  // routes accepted requests to the error leg.
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

} // namespace emel::io::read::events
