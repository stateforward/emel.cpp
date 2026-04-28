#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::encoder::whisper {

enum class error : emel::error::type {
  none = 0,
  model_invalid = 1,
  sample_rate = 2,
  channel_count = 3,
  pcm_shape = 4,
  output_capacity = 5,
  workspace_capacity = 6,
  unsupported_variant = 7,
  internal_error = 8,
};

}  // namespace emel::speech::encoder::whisper
