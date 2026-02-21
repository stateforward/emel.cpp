#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "emel/emel.h"
#include "emel/tokenizer/preprocessor/events.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::tokenizer::preprocessor::action {

struct context {
  const event::preprocess * request = nullptr;
  const emel::model::data::vocab * vocab = nullptr;
  std::string_view text = {};
  bool parse_special = false;
  size_t fragment_capacity = 0;
  size_t fragment_count = 0;
  special_token_cache special_cache = {};
  emel::model::data::tokenizer_pre bpe_pre_id =
      emel::model::data::tokenizer_pre::DEFAULT;
  std::vector<std::string> bpe_regex_exprs = {};
  std::vector<std::string> bpe_words = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::tokenizer::preprocessor::action
