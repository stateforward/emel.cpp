#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::recognizer {

enum class error : emel::error::type {
  none = 0,
  invalid_request = 1,
  model_invalid = 2,
  tokenizer_invalid = 3,
  unsupported_model = 4,
  uninitialized = 5,
  backend = 6,
  output_capacity = 7,
};

}  // namespace emel::speech::recognizer
