#pragma once

#include "emel/emel.h"
#include "emel/error/error.hpp"

namespace emel::memory::kv {

enum class error : emel::error::type {
  none = EMEL_OK,
  invalid_request = EMEL_ERR_INVALID_ARGUMENT,
  backend_error = EMEL_ERR_BACKEND,
  internal_error = EMEL_ERR_INTERNAL,
  out_of_memory = EMEL_ERR_OOM,
  untracked = EMEL_ERR_INTERNAL,
};

}  // namespace emel::memory::kv
