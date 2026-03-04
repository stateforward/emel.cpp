#pragma once

#include <cstdint>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/types.hpp"

namespace emel::text::encoders::rwkv::action {

struct context : emel::text::encoders::action::context {
  emel::text::encoders::detail::naive_trie token_matcher = {};
  bool rwkv_tables_ready = false;
  const emel::model::data::vocab *rwkv_vocab = nullptr;
};

}  // namespace emel::text::encoders::rwkv::action

namespace emel::text::encoders::rwkv::runtime {

struct encode_runtime {
  const emel::text::encoders::event::encode_runtime & event_;
  mutable int32_t unk_id = emel::text::encoders::detail::k_token_null;
  mutable bool unk_lookup_found = false;
};

}  // namespace emel::text::encoders::rwkv::runtime
