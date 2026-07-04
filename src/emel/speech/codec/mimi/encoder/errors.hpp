#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::codec::mimi::encoder {

enum class error : emel::error::type {
  none = 0,
  runtime_unbound = 1,
  request_shape = 2,
  buffer_capacity = 3,
};

} // namespace emel::speech::codec::mimi::encoder
