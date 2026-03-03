#pragma once

#include <array>
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

inline bool has_prefix(const std::string_view input,
                       const uint32_t pos,
                       const std::string_view prefix) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const size_t in_bounds = static_cast<size_t>(pos + prefix.size() <= size);
  const uint32_t safe_pos = pos * static_cast<uint32_t>(in_bounds);
  const size_t safe_size = prefix.size() * in_bounds;
  return in_bounds != 0u && input.substr(safe_pos, safe_size) == prefix;
}

inline uint32_t scan_quoted(const std::string_view input,
                            uint32_t pos,
                            const char terminator) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  ++pos;
  uint32_t matched = 0u;
  while (pos < size && matched == 0u) {
    const char c = input[pos];
    const size_t escaped = static_cast<size_t>(c == '\\' && pos + 1u < size);
    pos += static_cast<uint32_t>(escaped + 1u);
    matched = static_cast<uint32_t>(static_cast<size_t>(c == terminator) & (1u - escaped));
  }
  return pos;
}

inline uint32_t scan_braced_quantifier(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  ++pos;
  while (pos < size && input[pos] != '}') {
    ++pos;
  }
  pos += static_cast<uint32_t>(pos < size && input[pos] == '}');
  return pos;
}

inline uint32_t scan_token_ref(const std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  pos += static_cast<uint32_t>(input[pos] == '!');
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

inline size_t symbol_mode(const char c) noexcept {
  const size_t is_alternation = static_cast<size_t>(c == '|');
  const size_t is_dot = static_cast<size_t>(c == '.');
  const size_t is_open_group = static_cast<size_t>(c == '(');
  const size_t is_close_group = static_cast<size_t>(c == ')');
  const size_t is_simple_quantifier =
      static_cast<size_t>(c == '+' || c == '*' || static_cast<unsigned char>(c) == 63u);
  const size_t is_string_literal = static_cast<size_t>(c == '"');
  const size_t is_character_class = static_cast<size_t>(c == '[');
  const size_t is_braced_quantifier = static_cast<size_t>(c == '{');
  return is_alternation * 1u + is_dot * 2u + is_open_group * 3u + is_close_group * 4u +
         is_simple_quantifier * 5u + is_string_literal * 6u + is_character_class * 7u +
         is_braced_quantifier * 8u;
}

inline bool starts_rule_reference(const std::string_view input, const uint32_t start) noexcept {
  const char c = input[start];
  return c == '<' || (c == '!' && has_prefix(input, start + 1u, "<["));
}

inline event::token_kind symbol_kind(const size_t mode) noexcept {
  constexpr std::array<event::token_kind, 9> kinds = {
      event::token_kind::unknown,
      event::token_kind::alternation,
      event::token_kind::dot,
      event::token_kind::open_group,
      event::token_kind::close_group,
      event::token_kind::quantifier,
      event::token_kind::string_literal,
      event::token_kind::character_class,
      event::token_kind::quantifier,
  };
  return kinds[mode];
}

}  // namespace emel::gbnf::rule_parser::lexer::detail
