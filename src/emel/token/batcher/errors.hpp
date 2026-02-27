#pragma once

#include "emel/error/error.hpp"

namespace emel::token::batcher {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = 1u,
  backend_error = 6u,
  internal_error = 7u,
  untracked = 8u,
};

}  // namespace emel::token::batcher
