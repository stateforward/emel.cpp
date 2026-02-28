#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/preprocessor/errors.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::preprocessor::events {
struct preprocess_done;
struct preprocess_error;
}  // namespace emel::text::tokenizer::preprocessor::events

namespace emel::text::tokenizer::preprocessor::event {

struct preprocess {
  preprocess(const emel::model::data::vocab & vocab_ref,
             const std::string_view text_view,
             const bool parse_special_tokens,
             const std::span<fragment> fragments,
             size_t & fragment_count_ref,
             int32_t & error_ref) noexcept
      : vocab(vocab_ref),
        text(text_view),
        parse_special(parse_special_tokens),
        fragments_out(fragments),
        fragment_count_out(fragment_count_ref),
        error_out(error_ref) {}

  const emel::model::data::vocab & vocab;
  std::string_view text = {};
  bool parse_special = false;
  std::span<fragment> fragments_out = {};
  size_t & fragment_count_out;
  bool * preprocessed_out = nullptr;
  int32_t & error_out;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::preprocess_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::preprocess_error &) = nullptr;
};

struct preprocess_ctx {
  size_t fragment_count = 0;
  bool preprocessed = false;
  preprocessor::error phase_error = preprocessor::error::none;
  preprocessor::error err = preprocessor::error::none;
  bool result = false;
};

struct preprocess_runtime {
  const preprocess & request;
  preprocess_ctx & ctx;
};

}  // namespace emel::text::tokenizer::preprocessor::event

namespace emel::text::tokenizer::preprocessor::events {

struct preprocess_done {
  const event::preprocess * request = nullptr;
  size_t fragment_count = 0;
};

struct preprocess_error {
  const event::preprocess * request = nullptr;
  preprocessor::error err = preprocessor::error::none;
};

}  // namespace emel::text::tokenizer::preprocessor::events
