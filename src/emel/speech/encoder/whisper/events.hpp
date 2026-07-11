#pragma once

#include "emel/error/error.hpp"
#include "emel/speech/encoder/events.hpp"
#include "emel/speech/encoder/whisper/errors.hpp"

namespace emel::speech::encoder::whisper::events {

// The whisper variant consumes the component-level event vocabulary directly;
// the component contract carries the variant-bound dimensions.
using encode_done = emel::speech::encoder::events::encode_done;
using encode_error = emel::speech::encoder::events::encode_error;

} // namespace emel::speech::encoder::whisper::events

namespace emel::speech::encoder::whisper::event {

using encode = emel::speech::encoder::event::encode;

struct encode_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct encode_run {
  const encode &request;
  encode_ctx &ctx;
};

} // namespace emel::speech::encoder::whisper::event
