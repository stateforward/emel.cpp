#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::codec::mimi {

enum class error : emel::error::type {
  none = 0,
  not_initialized = 1,
  bind_failed = 2,
  latent_capacity = 3,
  encode_failed = 4,
  quantize_failed = 5,
  dequantize_failed = 6,
  decode_failed = 7,
};

} // namespace emel::speech::codec::mimi
