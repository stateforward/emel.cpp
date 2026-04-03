#pragma once

#include "emel/error/error.hpp"

namespace emel::model::tensor {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  internal_error = (1u << 1),
};

}  // namespace emel::model::tensor
