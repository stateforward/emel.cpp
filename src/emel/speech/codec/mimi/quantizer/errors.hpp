#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::codec::mimi::quantizer {

enum class error : emel::error::type {
  none = 0,
  runtime_unbound = 1,
  request_shape = 2,
  buffer_capacity = 3,
  // a decode code addresses no codebook entry
  code_range = 4,
};

} // namespace emel::speech::codec::mimi::quantizer
