#pragma once

#include "emel/text/tokenizer/preprocessor/actions.hpp"

namespace emel::text::tokenizer::preprocessor::plamo2::action {

using emel::text::tokenizer::preprocessor::action::begin_preprocess;
using emel::text::tokenizer::preprocessor::action::build_specials;
using emel::text::tokenizer::preprocessor::action::clear_request;
using emel::text::tokenizer::preprocessor::action::context;
using emel::text::tokenizer::preprocessor::action::ensure_last_error;
using emel::text::tokenizer::preprocessor::action::mark_done;
using emel::text::tokenizer::preprocessor::action::on_unexpected;
using emel::text::tokenizer::preprocessor::action::partition_non_bpe;
using emel::text::tokenizer::preprocessor::action::reject_invalid;

}  // namespace emel::text::tokenizer::preprocessor::plamo2::action
