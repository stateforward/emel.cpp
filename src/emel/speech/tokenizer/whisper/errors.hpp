#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::tokenizer::whisper {

enum class error : emel::error::type {
  none = 0,
  tokenizer_json_invalid = 1,
  internal_error = 2,
  token_ids_invalid = 3,
  decode_policy_unsupported = 4,
};

} // namespace emel::speech::tokenizer::whisper
