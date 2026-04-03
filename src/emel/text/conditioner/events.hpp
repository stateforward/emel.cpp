#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/events.hpp"

namespace emel::text::conditioner::events {

struct binding_done;
struct binding_error;
struct conditioning_done;
struct conditioning_error;

}  // namespace emel::text::conditioner::events

namespace emel::text::conditioner::event {

struct bind {
  explicit bind(const emel::model::data::vocab & vocab_ref) noexcept
      : vocab(vocab_ref) {}

  const emel::model::data::vocab & vocab;
  emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  emel::text::encoders::encoder_kind encoder_variant =
      emel::text::encoders::encoder_kind::fallback;
  void * tokenizer_sm = nullptr;
  bool (*dispatch_tokenizer_bind)(void * tokenizer_sm,
                                  const emel::text::tokenizer::event::bind &) =
      nullptr;
  bool (*dispatch_tokenizer_tokenize)(
      void * tokenizer_sm,
      const emel::text::tokenizer::event::tokenize &) = nullptr;
  void * formatter_ctx = nullptr;
  emel::text::formatter::format_fn format_prompt =
      emel::text::formatter::format_raw;
  emel::text::formatter::contract_kind formatter_contract =
      emel::text::formatter::contract_kind::raw;
  bool add_special = true;
  bool parse_special = false;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::binding_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::binding_error &) = nullptr;
};

struct prepare {
  prepare(int32_t & token_count_out_ref, int32_t & error_out_ref) noexcept
      : token_count_out(token_count_out_ref),
        error_out(error_out_ref) {}

  std::span<const emel::text::formatter::chat_message> messages = {};
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  bool add_special = true;
  bool parse_special = false;
  bool use_bind_defaults = true;
  int32_t * token_ids_out = nullptr;
  int32_t token_capacity = 0;
  int32_t & token_count_out;
  int32_t & error_out;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::conditioning_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::conditioning_error &) = nullptr;
};

struct bind_ctx {
  error err = error::none;
  bool result = false;
  bool bind_accepted = false;
  int32_t bind_err_code = 0;
};

struct bind_runtime {
  const bind & request;
  bind_ctx & ctx;
};

struct prepare_ctx {
  error err = error::none;
  char * formatted = nullptr;
  size_t formatted_capacity = 0;
  size_t formatted_length = 0;
  bool add_special = true;
  bool parse_special = false;
  int32_t token_count = 0;
  bool result = false;
  bool format_accepted = false;
  int32_t format_err_code = 0;
  bool tokenize_accepted = false;
  int32_t tokenize_err_code = 0;
};

struct prepare_runtime {
  const prepare & request;
  prepare_ctx & ctx;
};

}  // namespace emel::text::conditioner::event

namespace emel::text::conditioner::events {

struct binding_done {
  const event::bind * request = nullptr;
};

struct binding_error {
  const event::bind * request = nullptr;
  int32_t err = 0;
};

struct conditioning_done {
  const event::prepare * request = nullptr;
  int32_t token_count = 0;
};

struct conditioning_error {
  const event::prepare * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::text::conditioner::events
