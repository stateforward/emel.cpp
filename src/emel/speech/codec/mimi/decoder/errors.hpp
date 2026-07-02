#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::codec::mimi::decoder {

enum class error : emel::error::type {
  none = 0,
  runtime_unbound = 1,
  request_shape = 2,
  buffer_capacity = 3,
  upsample_failed = 4,
  transformer_failed = 5,
  backend_failed = 6,
};

} // namespace emel::speech::codec::mimi::decoder
