#pragma once

#include "emel/error/error.hpp"

namespace emel::speech::generator::moshi::personaplex::session::error {

enum code : emel::error::type {
  none = 0,
  invalid_request,
  memory_initialize_failed,
  codec_initialize_failed,
  executor_initialize_failed,
  generator_initialize_failed,
  voice_load_failed,
  voice_prefill_failed,
  prompt_begin_failed,
  prompt_prefill_failed,
  encode_failed,
  generate_failed,
  decode_failed,
  unexpected_event,
};

} // namespace emel::speech::generator::moshi::personaplex::session::error
