#pragma once

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/types.hpp"

namespace emel::text::encoders::rwkv::action {

struct context : emel::text::encoders::action::context {
  emel::text::encoders::detail::naive_trie token_matcher = {};
  bool rwkv_tables_ready = false;
  const emel::model::data::vocab *rwkv_vocab = nullptr;
};

}  // namespace emel::text::encoders::rwkv::action
