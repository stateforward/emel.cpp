#pragma once

#include "emel/error/error.hpp"
#include "emel/speech/tokenizer/events.hpp"
#include "emel/speech/tokenizer/whisper/errors.hpp"

namespace emel::speech::tokenizer::whisper::events {

// The whisper variant consumes the component-level event vocabulary directly.
using detokenize_done = emel::speech::tokenizer::events::detokenize_done;
using detokenize_error = emel::speech::tokenizer::events::detokenize_error;

} // namespace emel::speech::tokenizer::whisper::events

namespace emel::speech::tokenizer::whisper::event {

using detokenize = emel::speech::tokenizer::event::detokenize;

struct detokenize_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t transcript_size = 0;
};

struct detokenize_run {
  const detokenize &request;
  detokenize_ctx &ctx;
};

} // namespace emel::speech::tokenizer::whisper::event
