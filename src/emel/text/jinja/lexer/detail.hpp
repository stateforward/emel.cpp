#pragma once

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"

namespace emel::text::jinja::lexer::detail {

inline constexpr int32_t error_code(const parser::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline void normalize_source(std::string &source) {
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
  if (!source.empty() && source.back() == '\n') {
    source.pop_back();
  }
}

inline bool is_word(const char ch) noexcept {
  return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

inline bool is_integer(const char ch) noexcept {
  return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

inline bool is_space(const char ch) noexcept {
  return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

inline void string_lstrip(std::string &s, const char *chars) {
  const size_t start = s.find_first_not_of(chars);
  if (start == std::string::npos) {
    s.clear();
    return;
  }
  s.erase(0, start);
}

inline void string_rstrip(std::string &s, const char *chars) {
  const size_t end = s.find_last_not_of(chars);
  if (end == std::string::npos) {
    s.clear();
    return;
  }
  s.erase(end + 1);
}

inline bool next_pos_is(const std::string_view source, const size_t pos,
                        const std::initializer_list<char> chars,
                        const size_t n = 1) noexcept {
  const size_t idx = pos + n;
  if (idx >= source.size()) {
    return false;
  }
  for (const char c : chars) {
    if (source[idx] == c) {
      return true;
    }
  }
  return false;
}

inline bool decode_escape(const char ch, char &out) noexcept {
  switch (ch) {
  case 'n':
    out = '\n';
    return true;
  case 't':
    out = '\t';
    return true;
  case 'r':
    out = '\r';
    return true;
  case 'b':
    out = '\b';
    return true;
  case 'f':
    out = '\f';
    return true;
  case 'v':
    out = '\v';
    return true;
  case '\\':
    out = '\\';
    return true;
  case '\'':
    out = '\'';
    return true;
  case '"':
    out = '"';
    return true;
  default:
    return false;
  }
}

inline bool is_closing_block(const std::string_view source,
                             const size_t pos) noexcept {
  return pos < source.size() && source[pos] == '-' &&
         next_pos_is(source, pos, {'%', '}'});
}

inline bool unary_prefix_allowed(const token_type last) noexcept {
  switch (last) {
  case token_type::identifier:
  case token_type::numeric_literal:
  case token_type::string_literal:
  case token_type::close_paren:
  case token_type::close_square_bracket:
    return false;
  default:
    return true;
  }
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

struct scan_plan {
  std::string source = {};
  std::vector<scan_outcome> outcomes = {};
};

inline bool at_text_boundary(const token_type type) noexcept {
  return type == token_type::close_statement ||
         type == token_type::close_expression || type == token_type::comment;
}

inline ::emel::text::jinja::lexer::cursor
emit_cursor(const ::emel::text::jinja::lexer::cursor &cursor,
            const size_t next_offset, const token_type type,
            const std::string_view token_text) noexcept {
  ::emel::text::jinja::lexer::cursor next = cursor;
  next.offset = next_offset;
  next.token_index = cursor.token_index + 1;
  next.last_token_type = type;
  next.last_block_rstrip = false;
  next.last_block_can_trim_newline = false;

  if (type == token_type::open_expression) {
    next.curly_bracket_depth = 0;
  } else if (type == token_type::open_curly_bracket) {
    next.curly_bracket_depth = cursor.curly_bracket_depth + 1;
  } else if (type == token_type::close_curly_bracket) {
    next.curly_bracket_depth =
        cursor.curly_bracket_depth > 0 ? cursor.curly_bracket_depth - 1 : 0;
  }

  if (type == token_type::close_statement ||
      type == token_type::close_expression) {
    next.last_block_can_trim_newline = true;
    next.last_block_rstrip = token_text.size() >= 3 && token_text[0] == '-' &&
                             token_text.back() == '}';
  } else if (type == token_type::comment) {
    next.last_block_can_trim_newline = true;
  }

  return next;
}

inline void set_error(scan_outcome &out, const size_t pos) noexcept {
  out.err = error_code(parser::error::parse_failed);
  out.error_pos = pos;
  out.has_token = false;
}

inline std::string consume_escaped_until(const std::string_view source,
                                         size_t &pos, const char terminal,
                                         scan_outcome &out) {
  std::string value;
  while (pos < source.size() && source[pos] != terminal) {
    if (source[pos] != '\\') {
      value.push_back(source[pos]);
      ++pos;
      continue;
    }

    ++pos;
    if (pos >= source.size()) {
      set_error(out, pos);
      return value;
    }

    char decoded = '\0';
    const char escaped = source[pos];
    if (!decode_escape(escaped, decoded)) {
      set_error(out, pos);
      return value;
    }
    value.push_back(decoded);
    ++pos;
  }
  return value;
}

inline std::string consume_numeric(const std::string_view source, size_t &pos) {
  std::string value;
  while (pos < source.size() && is_integer(source[pos])) {
    value.push_back(source[pos]);
    ++pos;
  }
  if (pos < source.size() && source[pos] == '.' && pos + 1 < source.size() &&
      is_integer(source[pos + 1])) {
    value.push_back(source[pos]);
    ++pos;
    while (pos < source.size() && is_integer(source[pos])) {
      value.push_back(source[pos]);
      ++pos;
    }
  }
  return value;
}

inline scan_outcome
scan_next_token(const ::emel::text::jinja::lexer::cursor &cursor) {
  scan_outcome out{};
  out.next_cursor = cursor;

  const std::string_view source = cursor.source;
  const size_t size = source.size();
  size_t pos = cursor.offset;

  while (pos < size) {
    if (at_text_boundary(cursor.last_token_type)) {
      const size_t start = pos;
      size_t end = start;
      while (pos < size && !(source[pos] == '{' &&
                             next_pos_is(source, pos, {'%', '{', '#'}))) {
        end = ++pos;
      }

      if (pos < size && source[pos] == '{' &&
          next_pos_is(source, pos, {'%', '#', '-'})) {
        size_t current = end;
        while (current > start) {
          const char c = source[current - 1];
          if (current == 1) {
            end = 0;
            break;
          }
          if (c == '\n') {
            end = current;
            break;
          }
          if (!is_space(c)) {
            break;
          }
          --current;
        }
      }

      std::string text = std::string(source.substr(start, end - start));
      if (cursor.last_block_can_trim_newline && !text.empty() &&
          text.front() == '\n') {
        text.erase(text.begin());
      }
      if (cursor.last_block_rstrip) {
        string_lstrip(text, " \t\r\n");
      }

      const bool is_lstrip_block = pos < size && source[pos] == '{' &&
                                   next_pos_is(source, pos, {'{', '%', '#'}) &&
                                   next_pos_is(source, pos, {'-'}, 2);
      if (is_lstrip_block) {
        string_rstrip(text, " \t\r\n");
      }

      if (!text.empty()) {
        out.has_token = true;
        out.token_value = token{token_type::text, std::move(text), start};
        out.next_cursor = emit_cursor(cursor, pos, out.token_value.type,
                                      out.token_value.value);
        return out;
      }
    }

    if (source[pos] == '{' && next_pos_is(source, pos, {'#'})) {
      const size_t start = pos;
      pos += 2;
      std::string comment;
      while (pos < size &&
             !(source[pos] == '#' && next_pos_is(source, pos, {'}'}))) {
        if (pos + 2 >= size) {
          set_error(out, pos);
          return out;
        }
        comment.push_back(source[pos]);
        ++pos;
      }
      if (pos + 1 >= size) {
        set_error(out, pos);
        return out;
      }
      pos += 2;
      out.has_token = true;
      out.token_value = token{token_type::comment, std::move(comment), start};
      out.next_cursor =
          emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
      return out;
    }

    if (source[pos] == '-' &&
        (cursor.last_token_type == token_type::open_expression ||
         cursor.last_token_type == token_type::open_statement)) {
      ++pos;
      if (pos >= size) {
        out.next_cursor = cursor;
        out.next_cursor.offset = pos;
        return out;
      }
    }

    while (pos < size && is_space(source[pos])) {
      ++pos;
    }
    if (pos >= size) {
      out.next_cursor = cursor;
      out.next_cursor.offset = pos;
      return out;
    }

    const char ch = source[pos];
    if (!is_closing_block(source, pos) && (ch == '-' || ch == '+')) {
      if (cursor.last_token_type == token_type::text ||
          cursor.last_token_type == token_type::eof) {
        set_error(out, pos);
        return out;
      }
      if (unary_prefix_allowed(cursor.last_token_type)) {
        const size_t start = pos;
        ++pos;
        std::string num = consume_numeric(source, pos);
        std::string value;
        value.reserve(num.size() + 1);
        value.push_back(ch);
        value += num;
        const token_type type = num.empty() ? token_type::unary_operator
                                            : token_type::numeric_literal;
        out.has_token = true;
        out.token_value = token{type, std::move(value), start};
        out.next_cursor = emit_cursor(cursor, pos, out.token_value.type,
                                      out.token_value.value);
        return out;
      }
    }

    for (const auto &entry : k_mapping_table) {
      if (entry.seq == "}}" && cursor.curly_bracket_depth > 0) {
        continue;
      }
      if (pos + entry.seq.size() <= size &&
          source.compare(pos, entry.seq.size(), entry.seq) == 0) {
        out.has_token = true;
        out.token_value = token{entry.type, std::string(entry.seq), pos};
        out.next_cursor =
            emit_cursor(cursor, pos + entry.seq.size(), out.token_value.type,
                        out.token_value.value);
        return out;
      }
    }

    if (ch == '\'' || ch == '"') {
      const size_t start = pos;
      ++pos;
      std::string value = consume_escaped_until(source, pos, ch, out);
      if (out.err != error_code(parser::error::none)) {
        return out;
      }
      if (pos >= size) {
        set_error(out, pos);
        return out;
      }
      ++pos;
      out.has_token = true;
      out.token_value =
          token{token_type::string_literal, std::move(value), start};
      out.next_cursor =
          emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
      return out;
    }

    if (is_integer(ch)) {
      const size_t start = pos;
      std::string value = consume_numeric(source, pos);
      out.has_token = true;
      out.token_value =
          token{token_type::numeric_literal, std::move(value), start};
      out.next_cursor =
          emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
      return out;
    }

    if (is_word(ch)) {
      const size_t start = pos;
      std::string value;
      while (pos < size && is_word(source[pos])) {
        value.push_back(source[pos]);
        ++pos;
      }
      out.has_token = true;
      out.token_value = token{token_type::identifier, std::move(value), start};
      out.next_cursor =
          emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
      return out;
    }

    set_error(out, pos);
    return out;
  }

  out.has_token = false;
  out.next_cursor = cursor;
  out.next_cursor.offset = pos;
  return out;
}

inline scan_outcome
scan_next_token_safe(const ::emel::text::jinja::lexer::cursor &cursor) {
  const bool invalid_source =
      cursor.source.data() == nullptr && !cursor.source.empty();
  const bool invalid_offset = cursor.offset > cursor.source.size();
  return (invalid_source || invalid_offset) ? scan_outcome{}
                                            : scan_next_token(cursor);
}

inline scan_plan build_scan_plan(const std::string_view source_text) {
  scan_plan plan{};
  plan.source = std::string(source_text);
  normalize_source(plan.source);

  ::emel::text::jinja::lexer::cursor cursor{
      plan.source,
      0,
      0,
      0,
      ::emel::text::jinja::token_type::close_statement,
      false,
      false,
  };

  for (;;) {
    const scan_outcome scan = scan_next_token_safe(cursor);
    plan.outcomes.push_back(scan);
    const bool terminal =
        scan.err != error_code(parser::error::none) || !scan.has_token;
    if (terminal) {
      break;
    }
    cursor = scan.next_cursor;
  }

  return plan;
}

} // namespace emel::text::jinja::lexer::detail
