#pragma once

#include "emel/text/tokenizer/bpe/split.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::preprocessor::action {

struct context {
  special_token_cache special_cache = {};
  emel::text::tokenizer::bpe::detail::split_scratch bpe_scratch = {};
};

}  // namespace emel::text::tokenizer::preprocessor::action
