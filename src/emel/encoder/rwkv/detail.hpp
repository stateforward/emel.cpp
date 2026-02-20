#pragma once

#include "emel/encoder/rwkv/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::rwkv::detail {

using emel::encoder::detail::encode_result;

inline encode_result encode_rwkv(const event::encode &ev,
                                 emel::encoder::action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);
  emel::encoder::detail::encode_bytes(ev, ctx, vocab,
                                      emel::encoder::detail::tokenizer_model::rwkv,
                                      result);
  return result;
}

}  // namespace emel::encoder::rwkv::detail
