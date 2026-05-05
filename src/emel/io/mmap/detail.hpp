#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"

namespace emel::io::mmap::detail {

struct map_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  uint32_t reserved_slot = k_invalid_mapping_handle;
  intptr_t os_resource = -1;
  void *mapped_base = nullptr;
  uint64_t mapped_bytes = 0u;
  uint64_t file_size_bytes = 0u;
  bool file_open_ok = false;
  bool file_size_ok = false;
  bool mapping_ok = false;
};

struct release_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  uint32_t target_slot = k_invalid_mapping_handle;
  void *unmap_base = nullptr;
  uint64_t unmap_bytes = 0u;
  intptr_t os_resource = -1;
  bool unmap_ok = false;
};

struct map_tensor_runtime {
  const event::map_tensor &request;
  map_attempt_status &status;
};

struct release_mapping_runtime {
  const event::release_mapping &request;
  release_attempt_status &status;
};

} // namespace emel::io::mmap::detail
