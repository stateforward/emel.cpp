#pragma once

#include <cstddef>

#include "emel/model/data.hpp"
#include "emel/text/encoders/any.hpp"
#include "emel/text/tokenizer/preprocessor/any.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::action {

constexpr size_t k_max_fragments =
    emel::text::tokenizer::preprocessor::k_max_fragments;
using encoder_kind = emel::text::encoders::encoder_kind;
using fragment_kind = emel::text::tokenizer::preprocessor::fragment_kind;
using fragment = emel::text::tokenizer::preprocessor::fragment;
using preprocessor_kind =
    emel::text::tokenizer::preprocessor::preprocessor_kind;

struct context {
  emel::text::tokenizer::preprocessor::any preprocessor_any = {};
  emel::text::encoders::any encoder_any = {};
  const emel::model::data::vocab *vocab = nullptr;
  bool is_bound = false;
  preprocessor_kind preprocess_kind = preprocessor_kind::fallback;
  encoder_kind model_kind = encoder_kind::fallback;

  context();
};

} // namespace emel::text::tokenizer::action
