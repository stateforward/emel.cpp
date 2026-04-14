#pragma once

#include "emel/error/error.hpp"

namespace emel::embeddings::generator {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  model_invalid = (1u << 1),
  backend = (1u << 2),
};

}  // namespace emel::embeddings::generator
