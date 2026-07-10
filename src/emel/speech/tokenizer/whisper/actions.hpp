#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/speech/tokenizer/whisper/detail.hpp"
#include "emel/speech/tokenizer/whisper/events.hpp"

namespace emel::speech::tokenizer::whisper::action {

// The tokenizer machine holds no persistent actor-owned state.
struct context {};

struct effect_begin_detokenize {
  void operator()(const event::detokenize_run &, context &) const noexcept {}
};

struct effect_detokenize {
  void operator()(const event::detokenize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.transcript_size =
        static_cast<int32_t>(detail::decode_token_ids(
            runtime_ev.request.tokenizer_json, runtime_ev.request.token_ids,
            runtime_ev.request.transcript.data(),
            static_cast<uint64_t>(runtime_ev.request.transcript.size())));
    runtime_ev.request.transcript_size_out = runtime_ev.ctx.transcript_size;
    runtime_ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_mark_tokenizer_json_invalid {
  void operator()(const event::detokenize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = emel::error::cast(error::tokenizer_json_invalid);
  }
};

struct effect_store_error_out {
  void operator()(const event::detokenize_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::detokenize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::detokenize_done{
        .request = &runtime_ev.request,
        .transcript_size = runtime_ev.ctx.transcript_size,
    });
  }
};

struct effect_emit_error {
  void operator()(const event::detokenize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::detokenize_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class unexpected_ev>
  void operator()(const unexpected_ev &, context &) const noexcept {}
};

} // namespace emel::speech::tokenizer::whisper::action
