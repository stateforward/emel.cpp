#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/encoders/any.hpp"
#include "emel/text/tokenizer/errors.hpp"
#include "emel/text/tokenizer/preprocessor/any.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::events {
struct tokenizer_done;
struct tokenizer_error;
struct tokenizer_bind_done;
struct tokenizer_bind_error;
} // namespace emel::text::tokenizer::events

namespace emel::text::tokenizer::event {

struct bind {
  const emel::model::data::vocab *vocab = nullptr;
  emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  emel::text::encoders::encoder_kind encoder_variant =
      emel::text::encoders::encoder_kind::fallback;
  int32_t *error_out = nullptr;
  void *owner_sm = nullptr;
  bool (*dispatch_done)(void *owner_sm,
                        const events::tokenizer_bind_done &) = nullptr;
  bool (*dispatch_error)(void *owner_sm,
                         const events::tokenizer_bind_error &) = nullptr;
};

struct tokenize {
  const emel::model::data::vocab *vocab = nullptr;
  std::string_view text = {};
  bool add_special = false;
  bool parse_special = false;
  int32_t *token_ids_out = nullptr;
  int32_t token_capacity = 0;
  int32_t *token_count_out = nullptr;
  int32_t *error_out = nullptr;
  void *owner_sm = nullptr;
  bool (*dispatch_done)(void *owner_sm,
                        const events::tokenizer_done &) = nullptr;
  bool (*dispatch_error)(void *owner_sm,
                         const events::tokenizer_error &) = nullptr;
};

struct bind_ctx {
  int32_t err = error_code(error::none);
  bool result = false;
};

struct tokenize_ctx {
  std::array<emel::text::tokenizer::preprocessor::fragment,
             emel::text::tokenizer::preprocessor::k_max_fragments>
      fragments = {};
  size_t fragment_count = 0;
  size_t fragment_index = 0;
  bool preprocessed = false;
  bool preprocess_accepted = false;
  int32_t preprocess_err_code = error_code(error::none);
  bool encode_accepted = false;
  int32_t encode_err_code = error_code(error::none);
  int32_t encode_token_count = 0;
  int32_t token_count = 0;
  int32_t err = error_code(error::none);
  bool result = false;
};

struct bind_runtime {
  const bind &request;
  bind_ctx &ctx;
};

struct tokenize_runtime {
  const tokenize &request;
  tokenize_ctx &ctx;
};

} // namespace emel::text::tokenizer::event

namespace emel::text::tokenizer::events {

struct tokenizer_done {
  const event::tokenize *request = nullptr;
  int32_t token_count = 0;
};

struct tokenizer_error {
  const event::tokenize *request = nullptr;
  int32_t err = 0;
};

struct tokenizer_bind_done {
  const event::bind *request = nullptr;
};

struct tokenizer_bind_error {
  const event::bind *request = nullptr;
  int32_t err = 0;
};

} // namespace emel::text::tokenizer::events
