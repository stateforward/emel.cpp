#pragma once

#include <cstdint>

namespace emel::gbnf::term_parser::events {

enum class term_kind : uint8_t {
  unknown = 0,
  string_literal = 1,
  character_class = 2,
  rule_reference = 3,
  dot = 4,
  open_group = 5,
  close_group = 6,
  quantifier = 7,
  alternation = 8,
  newline = 9,
};

}  // namespace emel::gbnf::term_parser::events
