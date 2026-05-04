#pragma once

#include "emel/error/error.hpp"

namespace emel::io::loader {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  unsupported_strategy = (1u << 1),
  unavailable = (1u << 2),
  internal_error = (1u << 3),
  untracked = (1u << 4),
};

} // namespace emel::io::loader
