#pragma once

#include "emel/error/error.hpp"

namespace emel::text::tokenizer::needle {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  model_invalid = (1u << 1),
  capacity = (1u << 2),
  parse_failed = (1u << 3),
  internal_error = (1u << 4),
};

} // namespace emel::text::tokenizer::needle
