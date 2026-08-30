#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/needle/errors.hpp"

namespace emel::text::tokenizer::needle {

namespace events {

struct load_done;
struct load_error;

} // namespace events

namespace event {

using load_done_fn = emel::callback<void(const events::load_done &)>;
using load_error_fn = emel::callback<void(const events::load_error &)>;

// Loads the `.cact` trailing RAW SentencePiece-BPE dump into the shared
// text/tokenizer vocab so the existing SPM preprocessor + encoder machines
// consume it unchanged. `blob` is the zero-copy RAW tensor payload from the
// cact loader; `vocab_out` is caller-owned storage.
struct load {
  std::span<const uint8_t> blob = {};
  emel::model::data::vocab &vocab_out;
  const load_done_fn &on_done;
  const load_error_fn &on_error;

  load(std::span<const uint8_t> blob_in, emel::model::data::vocab &vocab_out_in,
       const load_done_fn &on_done_in,
       const load_error_fn &on_error_in) noexcept
      : blob(blob_in), vocab_out(vocab_out_in), on_done(on_done_in),
        on_error(on_error_in) {}
};

struct load_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct load_runtime {
  const load &request;
  load_ctx &ctx;
};

} // namespace event

namespace events {

struct load_done {
  const event::load &request;
  const emel::model::data::vocab &vocab_out;
};

struct load_error {
  const event::load &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace events

} // namespace emel::text::tokenizer::needle
