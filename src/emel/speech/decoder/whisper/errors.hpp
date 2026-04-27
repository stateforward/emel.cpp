#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::decoder::whisper {

enum class error : emel::error::type {
  none = 0,
  model_invalid = 1,
  encoder_state = 2,
  prompt_token = 3,
  logits_capacity = 4,
  transcript_capacity = 5,
  workspace_capacity = 6,
  unsupported_variant = 7,
  internal_error = 8,
  decode_policy = 9,
  generated_token_capacity = 10,
};

}  // namespace emel::speech::decoder::whisper
