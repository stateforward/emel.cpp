#pragma once

#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"
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
  const emel::model::data::vocab * vocab = nullptr;
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
  std::string_view input = {};
  bool add_special = true;
  bool parse_special = false;
  bool use_bind_defaults = true;
  int32_t * token_ids_out = nullptr;
  int32_t token_capacity = 0;
  int32_t * token_count_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::conditioning_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::conditioning_error &) = nullptr;
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
