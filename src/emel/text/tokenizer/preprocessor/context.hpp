#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/text/tokenizer/preprocessor/types.hpp"
#include "emel/text/tokenizer/bpe/split.hpp"

namespace emel::text::tokenizer::preprocessor::action {

struct context {
  const emel::model::data::vocab * vocab = nullptr;
  std::string_view text = {};
  fragment * fragments_out = nullptr;
  bool parse_special = false;
  bool preprocessed = false;
  size_t fragment_capacity = 0;
  size_t fragment_count = 0;
  special_token_cache special_cache = {};
  emel::text::tokenizer::bpe::detail::split_scratch bpe_scratch = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::text::tokenizer::preprocessor::action
