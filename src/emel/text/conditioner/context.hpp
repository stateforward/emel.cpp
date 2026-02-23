#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/events.hpp"

namespace emel::text::conditioner::action {

inline constexpr size_t k_max_formatted_bytes = 32768;

struct context {
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
  bool add_special_default = true;
  bool parse_special_default = false;
  bool is_bound = false;

  std::array<char, k_max_formatted_bytes> formatted = {};
  std::string_view input = {};
  size_t formatted_length = 0;
  bool add_special = true;
  bool parse_special = false;
  int32_t * token_ids_out = nullptr;
  int32_t token_capacity = 0;
  int32_t token_count = 0;

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::text::conditioner::action
