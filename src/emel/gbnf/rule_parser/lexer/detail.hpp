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
  uint32_t in_comment = 0u;
  uint32_t active = 1u;
  for (uint32_t scan = pos; scan < size; ++scan) {
    const char c = input[scan];
    const uint32_t is_space = static_cast<uint32_t>(c == ' ' || c == '\t');
    const uint32_t is_hash = static_cast<uint32_t>(c == '#');
    const uint32_t is_newline = static_cast<uint32_t>(is_newline_char(c));
    const uint32_t base_mode = 1u - in_comment;
    const uint32_t advance_space = active & base_mode & is_space;
    const uint32_t advance_hash = active & base_mode & is_hash;
    const uint32_t advance_comment = active & in_comment & (1u - is_newline);
    const uint32_t advanced = advance_space | advance_hash | advance_comment;
    pos += advanced;
    in_comment = ((in_comment | advance_hash) & (1u - is_newline));
    active &= advanced;
  }
  return pos;
}

inline uint32_t scan_token_ref_plain(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const uint32_t start = pos;
  const uint32_t has_open = static_cast<uint32_t>(pos + 1u < size && input[pos] == '<' &&
                                                   input[pos + 1u] == '[');
  const uint32_t digits_start = pos + (has_open * 2u);
  uint32_t digits_len = 0u;
  uint32_t active = has_open;
  for (uint32_t scan = digits_start; scan < size; ++scan) {
    const char c = input[scan];
    const uint32_t is_digit = static_cast<uint32_t>(c >= '0' && c <= '9');
    const uint32_t advance = active & is_digit;
    digits_len += advance;
    active = advance;
  }
  const uint32_t pos_after_digits = digits_start + digits_len;
  const uint32_t is_closed = has_open &
                             static_cast<uint32_t>(pos_after_digits + 1u < size &&
                                                   input[pos_after_digits] == ']' &&
                                                   input[pos_after_digits + 1u] == '>');
  const uint32_t end_positions[2] = {
      start, static_cast<uint32_t>(pos_after_digits + (is_closed * 2u))};
  return end_positions[has_open];
}

inline uint32_t token_start(const lexer::cursor & cursor) noexcept {
  return skip_layout(cursor.input, cursor.offset);
}

}  // namespace emel::gbnf::rule_parser::lexer::detail
