#pragma once

#include "emel/error/error.hpp"

namespace emel::text::renderer {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  backend_error = (1u << 1),
  model_invalid = (1u << 2),
  internal_error = (1u << 3),
  untracked = (1u << 4),
};

}  // namespace emel::text::renderer

