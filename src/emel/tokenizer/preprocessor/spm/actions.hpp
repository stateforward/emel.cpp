#pragma once

#include "emel/tokenizer/preprocessor/actions.hpp"

namespace emel::tokenizer::preprocessor::spm::action {

using emel::tokenizer::preprocessor::action::begin_preprocess;
using emel::tokenizer::preprocessor::action::build_specials;
using emel::tokenizer::preprocessor::action::clear_request;
using emel::tokenizer::preprocessor::action::context;
using emel::tokenizer::preprocessor::action::ensure_last_error;
using emel::tokenizer::preprocessor::action::mark_done;
using emel::tokenizer::preprocessor::action::on_unexpected;
using emel::tokenizer::preprocessor::action::partition_non_bpe;
using emel::tokenizer::preprocessor::action::reject_invalid;

}  // namespace emel::tokenizer::preprocessor::spm::action
