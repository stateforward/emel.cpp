#pragma once

#include "emel/error/error.hpp"

namespace emel::model::tensor {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  capacity = (1u << 1),
  backend_error = (1u << 2),
  model_invalid = (1u << 3),
  out_of_memory = (1u << 4),
  internal_error = (1u << 5),
  untracked = (1u << 6),
  io_mmap_unsupported = (1u << 7),
  io_mmap_failed = (1u << 8),
  tensor_already_resident = (1u << 9),
  tensor_unmapped = (1u << 10),
  io_read_unsupported = (1u << 11),
  io_read_failed = (1u << 12),
  io_staged_read_unsupported = (1u << 13),
  io_staged_read_failed = (1u << 14),
};

} // namespace emel::model::tensor
