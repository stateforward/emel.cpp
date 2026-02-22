#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/encoder/any.hpp"
#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::tokenizer::action {

constexpr size_t k_max_fragments =
    emel::tokenizer::preprocessor::k_max_fragments;
constexpr size_t k_max_special_tokens =
    emel::tokenizer::preprocessor::k_max_special_tokens;
using encoder_kind = emel::encoder::encoder_kind;

using fragment_kind = emel::tokenizer::preprocessor::fragment_kind;
using fragment = emel::tokenizer::preprocessor::fragment;
using special_token = emel::tokenizer::preprocessor::special_token;
using special_token_cache = emel::tokenizer::preprocessor::special_token_cache;

struct context {
  emel::encoder::any encoder_any = {};
  std::array<fragment, k_max_fragments> fragments = {};
  size_t fragment_count = 0;
  size_t fragment_index = 0;
  special_token_cache special_cache = {};
  const emel::model::data::vocab *vocab = nullptr;
  std::string_view text = {};
  bool add_special = false;
  bool parse_special = false;
  int32_t *token_ids_out = nullptr;
  int32_t token_capacity = 0;
  encoder_kind model_kind = encoder_kind::fallback;
  int32_t token_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  context();
};

} // namespace emel::tokenizer::action
