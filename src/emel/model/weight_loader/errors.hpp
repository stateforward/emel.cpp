#pragma once

#include "emel/error/error.hpp"

namespace emel::model::weight_loader {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  capacity = (1u << 1),
  backend_error = (1u << 2),
  model_invalid = (1u << 3),
  out_of_memory = (1u << 4),
  internal_error = (1u << 5),
  untracked = (1u << 6),
};

}  // namespace emel::model::weight_loader
