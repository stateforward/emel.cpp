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
  uint32_t scan_more = 1u;
  while (pos < size && scan_more != 0u) {
    const char c = input[pos];
    const size_t mode =
        static_cast<size_t>(c == ' ' || c == '\t') + (static_cast<size_t>(c == '#') * 2u);
    const uint32_t advance_space = static_cast<uint32_t>(mode == 1u);
    const uint32_t skip_comment = static_cast<uint32_t>(mode == 2u);
    pos += advance_space;
    pos += skip_comment;
    while (pos < size && skip_comment != 0u && !is_newline_char(input[pos])) {
      ++pos;
    }
    scan_more = advance_space | skip_comment;
  }
  return pos;
}

inline uint32_t scan_token_ref_plain(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const uint32_t start = pos;
  const uint32_t has_open = static_cast<uint32_t>(pos + 1u < size && input[pos] == '<' &&
                                                   input[pos + 1u] == '[');
  pos += has_open * 2u;
  while (pos < size && has_open != 0u && input[pos] >= '0' && input[pos] <= '9') {
    ++pos;
  }
  const uint32_t is_closed = has_open &
                             static_cast<uint32_t>(pos + 1u < size && input[pos] == ']' &&
                                                   input[pos + 1u] == '>');
  const uint32_t end_positions[2] = {start, static_cast<uint32_t>(pos + (is_closed * 2u))};
  return end_positions[has_open];
}

inline uint32_t token_start(const lexer::cursor & cursor) noexcept {
  return skip_layout(cursor.input, cursor.offset);
}

}  // namespace emel::gbnf::rule_parser::lexer::detail
