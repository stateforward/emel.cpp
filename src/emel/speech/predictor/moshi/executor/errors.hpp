#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::predictor::moshi::executor {

enum class error : emel::error::type {
  none = 0,
  not_initialized = 1,
  bind_failed = 2,
  request_shape = 3,
  model_mismatch = 4,
  graph_execution_unsupported = 5,
  unexpected_event = 6,
  reset_failed = 7,
};

} // namespace emel::speech::predictor::moshi::executor
