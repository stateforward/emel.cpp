#pragma once

#include "emel/encoder/context.hpp"
#include "emel/encoder/types.hpp"

namespace emel::encoder::rwkv::action {

struct context : emel::encoder::action::context {
  emel::encoder::detail::naive_trie token_matcher = {};
  bool rwkv_tables_ready = false;
  const emel::model::data::vocab *rwkv_vocab = nullptr;
};

}  // namespace emel::encoder::rwkv::action
