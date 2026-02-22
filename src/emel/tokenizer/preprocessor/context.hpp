#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/tokenizer/preprocessor/events.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"
#include "emel/tokenizer/bpe/split.hpp"

namespace emel::tokenizer::preprocessor::action {

struct context {
  const event::preprocess * request = nullptr;
  const emel::model::data::vocab * vocab = nullptr;
  std::string_view text = {};
  bool parse_special = false;
  size_t fragment_capacity = 0;
  size_t fragment_count = 0;
  special_token_cache special_cache = {};
  emel::tokenizer::bpe::detail::split_scratch bpe_scratch = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::tokenizer::preprocessor::action
