#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::codec::mimi {

enum class error : emel::error::type {
  none = 0,
  not_initialized = 1,
  // the model does not satisfy the mimi contract (validate_codec_contract)
  bind_failed = 2,
  // a caller arena is smaller than the required_* sizing contract, or the
  // model latent width exceeds the facade staging column
  arena_capacity = 3,
  // a frame request span disagrees with the bound runtime dimensions
  request_shape = 4,
  // a decode code addresses no codebook entry
  code_range = 5,
  // the event is not modeled in the current state (caller ordering error,
  // for example initialize while a session is ready or reset_stream before
  // initialization)
  unexpected_event = 6,
};

} // namespace emel::speech::codec::mimi
