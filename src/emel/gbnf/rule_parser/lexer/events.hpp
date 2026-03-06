#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"

namespace emel::gbnf::rule_parser::lexer::events {

struct next_done;
struct next_error;

}  // namespace emel::gbnf::rule_parser::lexer::events

namespace emel::gbnf::rule_parser::lexer {

struct cursor {
  std::string_view input = {};
  uint32_t offset = 0;
  uint32_t token_count = 0;
};

}  // namespace emel::gbnf::rule_parser::lexer

namespace emel::gbnf::rule_parser::lexer::event {

enum class token_kind : uint8_t {
  unknown = 0,
  identifier = 1,
  string_literal = 2,
  character_class = 3,
  rule_reference = 4,
  definition_operator = 5,
  alternation = 6,
  dot = 7,
  open_group = 8,
  close_group = 9,
  quantifier = 10,
  newline = 11,
};

struct token {
  token_kind kind = token_kind::unknown;
  std::string_view text = {};
  uint32_t start = 0;
  uint32_t end = 0;
};

struct next {
  const lexer::cursor & cursor;
  const callback<bool(const ::emel::gbnf::rule_parser::lexer::events::next_done &)> & on_done;
  const callback<bool(const ::emel::gbnf::rule_parser::lexer::events::next_error &)> & on_error;
};

struct scan_ctx {
  uint32_t start = 0;
  char first_char = '\0';
  bool has_input = false;
};

struct scan_next {
  const next & request;
  scan_ctx & ctx;
};

}  // namespace emel::gbnf::rule_parser::lexer::event

namespace emel::gbnf::rule_parser::lexer::events {

struct next_done {
  event::token token = {};
  bool has_token = false;
  lexer::cursor next_cursor = {};
};

struct next_error {
  int32_t err = 0;
};

}  // namespace emel::gbnf::rule_parser::lexer::events
