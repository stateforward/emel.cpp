#pragma once

#include "emel/error/error.hpp"

namespace emel::memory::streaming {

enum class error : emel::error::type {
  none = 0u,
  invalid_configuration = (1u << 0),
  uninitialized = (1u << 1),
  already_initialized = (1u << 2),
  position_overflow = (1u << 3),
  internal_error = (1u << 4),
};

} // namespace emel::memory::streaming
