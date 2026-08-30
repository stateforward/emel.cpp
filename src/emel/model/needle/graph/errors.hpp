#pragma once

#include "emel/error/error.hpp"

namespace emel::model::needle::graph {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  geometry_unsupported = (1u << 1),
  not_initialized = (1u << 2),
  capacity_exceeded = (1u << 3),
  kernel_rejected = (1u << 4),
  internal_error = (1u << 5),
};

} // namespace emel::model::needle::graph
