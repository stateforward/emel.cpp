#pragma once

#include "emel/error/error.hpp"

namespace emel::model::loader {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  parse_failed = (1u << 1),
  backend_error = (1u << 2),
  model_invalid = (1u << 3),
  internal_error = (1u << 4),
  untracked = (1u << 5),
};

}  // namespace emel::model::loader
