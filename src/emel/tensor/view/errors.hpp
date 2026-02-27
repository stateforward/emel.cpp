#pragma once

#include "emel/emel.h"
#include "emel/error/error.hpp"

namespace emel::tensor::view {

enum class error : emel::error::type {
  none = EMEL_OK,
  invalid_request = EMEL_ERR_INVALID_ARGUMENT,
  internal_error = EMEL_ERR_INTERNAL,
};

}  // namespace emel::tensor::view
