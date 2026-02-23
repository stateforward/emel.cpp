#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/text/encoders/any.hpp"
#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/events.hpp"
#include "emel/text/tokenizer/preprocessor/any.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::action {

constexpr size_t k_max_fragments =
    emel::text::tokenizer::preprocessor::k_max_fragments;
constexpr size_t k_max_special_tokens =
    emel::text::tokenizer::preprocessor::k_max_special_tokens;
using encoder_kind = emel::text::encoders::encoder_kind;

using fragment_kind = emel::text::tokenizer::preprocessor::fragment_kind;
using fragment = emel::text::tokenizer::preprocessor::fragment;
using preprocessor_kind = emel::text::tokenizer::preprocessor::preprocessor_kind;

struct context {
  emel::text::tokenizer::preprocessor::any preprocessor_any = {};
  emel::text::encoders::any encoder_any = {};
  std::array<fragment, k_max_fragments> fragments = {};
  size_t fragment_count = 0;
  size_t fragment_index = 0;
  const emel::model::data::vocab *vocab = nullptr;
  std::string_view text = {};
  bool add_special = false;
  bool parse_special = false;
  bool fragments_preprocessed = false;
  int32_t *token_ids_out = nullptr;
  int32_t token_capacity = 0;
  bool is_bound = false;
  preprocessor_kind preprocess_kind = preprocessor_kind::fallback;
  encoder_kind model_kind = encoder_kind::fallback;
  int32_t token_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  context();
};

} // namespace emel::text::tokenizer::action
