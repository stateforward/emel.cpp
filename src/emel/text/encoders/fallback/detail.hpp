#pragma once

#include "emel/text/encoders/fallback/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::fallback::detail {

using emel::text::encoders::detail::encode_result;

inline encode_result encode_fallback(const event::encode &ev,
                                     emel::text::encoders::action::context &ctx,
                                     const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::text::encoders::detail::ensure_tables(ctx);
  emel::text::encoders::detail::encode_bytes(ev, ctx, vocab,
                                      emel::model::data::tokenizer_model::UNKNOWN,
                                      result);
  return result;
}

}  // namespace emel::text::encoders::fallback::detail
