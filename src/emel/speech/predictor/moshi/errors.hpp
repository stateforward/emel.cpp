#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::predictor::moshi {

enum class error : emel::error::type {
  none = 0,
  not_initialized = 1,
  bind_failed = 2,
  memory = 3,
  request_shape = 4,
  graph_runtime_unavailable = 5,
  graph_runtime = 6,
  output_waiting = 7,
  voice_contract = 8,
  voice_prompt_pending = 9,
  personaplex_prompt = 10,
  unexpected_event = 11,
};

} // namespace emel::speech::predictor::moshi
