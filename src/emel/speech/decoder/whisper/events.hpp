#pragma once

#include "emel/error/error.hpp"
#include "emel/speech/decoder/events.hpp"
#include "emel/speech/decoder/whisper/errors.hpp"

namespace emel::speech::decoder::whisper::events {

// The whisper variant consumes the component-level event vocabulary directly;
// the component contract carries the variant-bound dimensions.
using decode_done = emel::speech::decoder::events::decode_done;
using decode_error = emel::speech::decoder::events::decode_error;

} // namespace emel::speech::decoder::whisper::events

namespace emel::speech::decoder::whisper::event {

using decode = emel::speech::decoder::event::decode;

struct decode_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct decode_run {
  const decode &request;
  decode_ctx &ctx;
};

} // namespace emel::speech::decoder::whisper::event
