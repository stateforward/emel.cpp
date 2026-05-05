#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/mmap/errors.hpp"

namespace emel::io::mmap::events {

struct map_tensor_done;
struct map_tensor_error;
struct release_mapping_done;
struct release_mapping_error;

} // namespace emel::io::mmap::events

namespace emel::io::mmap::event {

struct map_tensor_request {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  // The action layer copies this view into a bounded stack buffer before
  // calling platform C APIs; callers do not need to pass a null-terminated
  // view. Embedded NUL bytes are rejected by the file-path guard.
  std::string_view file_path = {};
};

struct map_tensor {
  const map_tensor_request &request;
  // Required for valid map requests because map_tensor_done carries the
  // handle needed to release the mapped resource.
  emel::callback<void(const events::map_tensor_done &)> on_done = {};
  emel::callback<void(const events::map_tensor_error &)> on_error = {};

  explicit map_tensor(const map_tensor_request &request_in) noexcept
      : request(request_in) {}
};

struct release_mapping {
  int32_t tensor_id = -1;
  uint32_t handle = k_invalid_mapping_handle;
  emel::callback<void(const events::release_mapping_done &)> on_done = {};
  emel::callback<void(const events::release_mapping_error &)> on_error = {};

  release_mapping(int32_t tensor_id_in, uint32_t handle_in) noexcept
      : tensor_id(tensor_id_in), handle(handle_in) {}
};

} // namespace emel::io::mmap::event

namespace emel::io::mmap::events {

struct map_tensor_done {
  const event::map_tensor &request;
  uint32_t handle = k_invalid_mapping_handle;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
};

struct map_tensor_error {
  const event::map_tensor &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct release_mapping_done {
  const event::release_mapping &request;
};

struct release_mapping_error {
  const event::release_mapping &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::io::mmap::events
