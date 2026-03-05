#pragma once

#include <cstdint>
#include <string_view>

#include "emel/gbnf/rule_parser/lexer/events.hpp"

namespace emel::gbnf::rule_parser::lexer::detail {

inline bool is_word_char(const char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' ||
         (c >= '0' && c <= '9');
}

inline bool is_newline_char(const char c) noexcept {
  return c == '\n' || c == '\r';
}

inline uint32_t skip_layout(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  while (pos < size) {
    const char c = input[pos];
    if (c == ' ' || c == '\t') {
      ++pos;
      continue;
    }
    if (c != '#') {
      break;
    }
    ++pos;
    while (pos < size && !is_newline_char(input[pos])) {
      ++pos;
    }
  }
  return pos;
}

inline uint32_t scan_token_ref_plain(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  if (pos + 1u >= size || input[pos] != '<' || input[pos + 1u] != '[') {
    return pos;
  }
  uint32_t scan = static_cast<uint32_t>(pos + 2u);
  while (scan < size) {
    const char c = input[scan];
    if (c < '0' || c > '9') {
      break;
    }
    ++scan;
  }
  if (scan + 1u < size && input[scan] == ']' && input[scan + 1u] == '>') {
    return static_cast<uint32_t>(scan + 2u);
  }
  return pos;
}

inline uint32_t token_start(const lexer::cursor & cursor) noexcept {
  return skip_layout(cursor.input, cursor.offset);
}

}  // namespace emel::gbnf::rule_parser::lexer::detail
