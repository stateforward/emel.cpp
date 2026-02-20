#pragma once

#include "emel/encoder/plamo2/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::plamo2::detail {

using emel::encoder::detail::encode_result;

inline encode_result encode_plamo2(const event::encode &ev,
                                   emel::encoder::action::context &ctx,
                                   const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);
  emel::encoder::detail::encode_bytes(ev, ctx, vocab,
                                      emel::encoder::detail::tokenizer_model::plamo2,
                                      result);
  return result;
}

}  // namespace emel::encoder::plamo2::detail
