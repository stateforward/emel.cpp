#pragma once

#include "emel/error/error.hpp"

namespace emel::tensor::view {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  internal_error = (1u << 1),
};

}  // namespace emel::tensor::view
