#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/events.hpp"

namespace emel::text::conditioner::action {

inline constexpr size_t k_max_formatted_bytes = 32768;

struct context {
  const emel::model::data::vocab * vocab = nullptr;
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
  bool add_special_default = true;
  bool parse_special_default = false;
  bool is_bound = false;
};

}  // namespace emel::text::conditioner::action
