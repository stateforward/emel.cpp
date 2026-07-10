#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::tokenizer::moshi {

enum class error : emel::error::type {
  none = 0u,
  invalid_configuration = (1u << 0),
  uninitialized = (1u << 1),
  already_initialized = (1u << 2),
  request_shape = (1u << 3),
  phase_order = (1u << 4),
  position_overflow = (1u << 5),
  internal_error = (1u << 6),
};

} // namespace emel::speech::tokenizer::moshi
