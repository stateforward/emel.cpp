#pragma once

#include "emel/encoder/context.hpp"
#include "emel/model/data.hpp"
#include "emel/tokenizer/bpe/split.hpp"

namespace emel::encoder::bpe::action {

struct context : emel::encoder::action::context {
  emel::tokenizer::bpe::detail::split_scratch bpe_scratch = {};
};

}  // namespace emel::encoder::bpe::action
