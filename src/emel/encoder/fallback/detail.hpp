#pragma once

#include "emel/encoder/fallback/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::fallback::detail {

using emel::encoder::detail::encode_result;

inline encode_result encode_fallback(const event::encode &ev,
                                     emel::encoder::action::context &ctx,
                                     const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);
  emel::encoder::detail::encode_bytes(ev, ctx, vocab,
                                      emel::encoder::detail::tokenizer_model::unknown,
                                      result);
  return result;
}

}  // namespace emel::encoder::fallback::detail
