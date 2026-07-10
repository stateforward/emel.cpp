#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::generator {

enum class error : emel::error::type {
  none = 0u,
  uninitialized = (1u << 0),
  already_initialized = (1u << 1),
  invalid_request = (1u << 2),
  cutover_pending = (1u << 3),
  internal_error = (1u << 4),
  memory_initialize_failed = (1u << 5),
  encoder_initialize_failed = (1u << 6),
  decoder_initialize_failed = (1u << 7),
  runtime_initialize_failed = (1u << 8),
  predictor_initialize_failed = (1u << 9),
  conditioning_failed = (1u << 10),
  encode_failed = (1u << 11),
  predict_failed = (1u << 12),
  decode_failed = (1u << 13),
};

} // namespace emel::speech::generator
