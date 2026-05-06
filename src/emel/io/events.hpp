#pragma once

#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"

namespace emel::io::event {

struct tensor_load_span {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  std::string_view file_path = {};
  const void *source_buffer = nullptr;
  uint64_t source_buffer_bytes = 0u;
  emel::error::type source_error = emel::error::type{0u};
  void *target = nullptr;
  uint64_t target_bytes = 0u;
};

} // namespace emel::io::event
