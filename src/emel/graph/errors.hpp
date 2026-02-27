#pragma once

#include "emel/error/error.hpp"

namespace emel::graph {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  assembler_failed = (1u << 1),
  processor_failed = (1u << 2),
  busy = (1u << 3),
  internal_error = (1u << 4),
  untracked = (1u << 5),
};

}  // namespace emel::graph
