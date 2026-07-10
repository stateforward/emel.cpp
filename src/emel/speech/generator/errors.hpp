#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::generator {

enum class error : emel::error::type {
  none = 0u,
  uninitialized = (1u << 0),
  already_initialized = (1u << 1),
  invalid_request = (1u << 2),
  cutover_pending = (1u << 3),
  internal_error = (1u << 4),
};

} // namespace emel::speech::generator
