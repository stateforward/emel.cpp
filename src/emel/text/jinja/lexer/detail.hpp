#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"

namespace emel::text::jinja::lexer::detail {

inline constexpr int32_t error_code(const parser::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline char view_char_at_or(const std::string_view source,
                            const size_t index,
                            const char fallback) noexcept {
  constexpr size_t k_fallback_index = 0u;
  const bool in_range = index < source.size();
  const size_t safe_index = select_size(in_range, index, k_fallback_index);
  const std::array<char, 1> fallback_buffer{fallback};
  const std::array<const char *, 2> data_ptrs{
      fallback_buffer.data(),
      source.data(),
  };
  return data_ptrs[static_cast<size_t>(in_range)][safe_index];
}

inline void normalize_source(std::string &source) {
  const bool has_cr = source.find('\r') != std::string::npos;
  if (!has_cr) {
    if (!source.empty() && source.back() == '\n') {
      source.pop_back();
    }
    return;
  }

  for (std::string::size_type pos = 0;
       (pos = source.find("\r\n", pos)) != std::string::npos;) {
    source.erase(pos, 1);
    ++pos;
  }
  for (std::string::size_type pos = 0;
       (pos = source.find('\r', pos)) != std::string::npos;) {
    source.replace(pos, 1, 1, '\n');
    ++pos;
  }
  using trim_handler_t = void (*)(std::string &) noexcept;
  const trim_handler_t trim_handlers[2] = {
      +[](std::string &) noexcept {},
      +[](std::string &value) noexcept { value.pop_back(); },
  };
  const bool has_trailing_newline = !source.empty() && source.back() == '\n';
  trim_handlers[static_cast<size_t>(has_trailing_newline)](source);
}

inline bool is_word(const char ch) noexcept {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
         (ch >= '0' && ch <= '9') || ch == '_';
}

inline bool is_integer(const char ch) noexcept {
  return ch >= '0' && ch <= '9';
}

inline bool is_space(const char ch) noexcept {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' ||
         ch == '\v';
}

inline void string_lstrip(std::string &s, const char *chars) {
  const size_t start = s.find_first_not_of(chars);
  const size_t erase_count = select_size(start == std::string::npos, s.size(), start);
  s.erase(0, erase_count);
}

inline void string_rstrip(std::string &s, const char *chars) {
  const size_t end = s.find_last_not_of(chars);
  const size_t keep_count = select_size(end == std::string::npos, 0u, end + 1u);
  s.erase(keep_count);
}

template <char... chars>
inline bool next_pos_is(const std::string_view source,
                        const size_t pos,
                        const size_t n = 1) noexcept {
  const size_t idx = pos + n;
  if (idx >= source.size()) {
    return false;
  }
  const char candidate = source[idx];
  return ((candidate == chars) || ...);
}

inline bool is_closing_block(const std::string_view source,
                             const size_t pos) noexcept {
  return pos < source.size() && source[pos] == '-' &&
         next_pos_is<'%', '}'>(source, pos);
}

inline bool unary_prefix_allowed(const token_type last) noexcept {
  const bool disallowed = last == token_type::identifier ||
      last == token_type::numeric_literal ||
      last == token_type::string_literal ||
      last == token_type::close_paren ||
      last == token_type::close_square_bracket;
  return !disallowed;
}

struct mapping {
  std::string_view seq = {};
  token_type type = token_type::eof;
};

inline constexpr std::array<mapping, 31> k_mapping_table{{
    {"{%-", token_type::open_statement},
    {"-%}", token_type::close_statement},
    {"{{-", token_type::open_expression},
    {"-}}", token_type::close_expression},
    {"{%", token_type::open_statement},
    {"%}", token_type::close_statement},
    {"{{", token_type::open_expression},
    {"}}", token_type::close_expression},
    {"(", token_type::open_paren},
    {")", token_type::close_paren},
    {"{", token_type::open_curly_bracket},
    {"}", token_type::close_curly_bracket},
    {"[", token_type::open_square_bracket},
    {"]", token_type::close_square_bracket},
    {",", token_type::comma},
    {".", token_type::dot},
    {":", token_type::colon},
    {"|", token_type::pipe},
    {"<=", token_type::comparison_binary_operator},
    {">=", token_type::comparison_binary_operator},
    {"==", token_type::comparison_binary_operator},
    {"!=", token_type::comparison_binary_operator},
    {"<", token_type::comparison_binary_operator},
    {">", token_type::comparison_binary_operator},
    {"+", token_type::additive_binary_operator},
    {"-", token_type::additive_binary_operator},
    {"~", token_type::additive_binary_operator},
    {"*", token_type::multiplicative_binary_operator},
    {"/", token_type::multiplicative_binary_operator},
    {"%", token_type::multiplicative_binary_operator},
    {"=", token_type::equals},
}};

struct scan_outcome {
  token token_value = {};
  bool has_token = false;
  ::emel::text::jinja::lexer::cursor next_cursor = {};
  int32_t err = error_code(parser::error::none);
  size_t error_pos = 0;
};

inline bool at_text_boundary(const token_type type) noexcept {
  return type == token_type::close_statement ||
         type == token_type::close_expression || type == token_type::comment;
}

inline ::emel::text::jinja::lexer::cursor
emit_cursor(const ::emel::text::jinja::lexer::cursor &cursor,
            const size_t next_offset,
            const token_type type,
            const std::string_view token_text) noexcept {
  ::emel::text::jinja::lexer::cursor next = cursor;
  next.offset = next_offset;
  next.token_index = cursor.token_index + 1;
  next.last_token_type = type;
  next.last_block_rstrip = false;
  next.last_block_can_trim_newline = false;
  const size_t is_open_expression = static_cast<size_t>(type == token_type::open_expression);
  const size_t is_open_curly = static_cast<size_t>(type == token_type::open_curly_bracket);
  const size_t is_close_curly = static_cast<size_t>(type == token_type::close_curly_bracket);
  const size_t can_pop_curly =
      is_close_curly * static_cast<size_t>(cursor.curly_bracket_depth > 0);
  const std::array<size_t, 2> depth_candidates = {
      cursor.curly_bracket_depth + is_open_curly - can_pop_curly,
      0u,
  };
  next.curly_bracket_depth = depth_candidates[static_cast<size_t>(is_open_expression != 0)];

  const bool closes_block = type == token_type::close_statement ||
      type == token_type::close_expression;
  const bool is_comment = type == token_type::comment;
  next.last_block_can_trim_newline = closes_block || is_comment;
  next.last_block_rstrip = closes_block && token_text.size() >= 3 &&
                           token_text[0] == '-' && token_text.back() == '}';

  return next;
}

inline void set_error(scan_outcome &out, const size_t pos) noexcept {
  out.err = error_code(parser::error::parse_failed);
  out.error_pos = pos;
  out.has_token = false;
}

inline void emit_no_token_cursor(scan_outcome &out,
                                 const ::emel::text::jinja::lexer::cursor &cursor,
                                 const size_t pos) noexcept {
  out.has_token = false;
  out.next_cursor = cursor;
  out.next_cursor.offset = pos;
}

inline void consume_fraction_none(std::string &, const std::string_view, size_t &) noexcept {}

inline void consume_fraction_some(std::string &value,
                                  const std::string_view source,
                                  size_t &pos) noexcept {
  value.push_back(source[pos]);
  ++pos;
  while (pos < source.size() && is_integer(source[pos])) {
    value.push_back(source[pos]);
    ++pos;
  }
}

inline std::string consume_numeric(const std::string_view source, size_t &pos) {
  std::string value;
  while (pos < source.size() && is_integer(source[pos])) {
    value.push_back(source[pos]);
    ++pos;
  }
  const bool has_fraction = pos < source.size() && source[pos] == '.' &&
      pos + 1 < source.size() && is_integer(source[pos + 1]);
  using fraction_handler_t = void (*)(std::string &, std::string_view, size_t &) noexcept;
  const fraction_handler_t fraction_handlers[2] = {
      consume_fraction_none,
      consume_fraction_some,
  };
  fraction_handlers[static_cast<size_t>(has_fraction)](value, source, pos);
  return value;
}

} // namespace emel::text::jinja::lexer::detail
