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
    {
    const size_t emel_branch_1 = static_cast<size_t>(!source.empty() && source.back() == '\n');
    for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 1u; emel_case_1 = 2u) {
            source.pop_back();
    }
    for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 0u; emel_case_1 = 2u) {

    }
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
    {
    const size_t emel_branch_2 = static_cast<size_t>(start == std::string::npos);
    for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 1u; emel_case_2 = 2u) {
            s.clear();
            return;
    }
    for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 0u; emel_case_2 = 2u) {

    }
  }
  s.erase(0, start);
}

inline void string_rstrip(std::string &s, const char *chars) {
  const size_t end = s.find_last_not_of(chars);
    {
    const size_t emel_branch_3 = static_cast<size_t>(end == std::string::npos);
    for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 1u; emel_case_3 = 2u) {
            s.clear();
            return;
    }
    for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 0u; emel_case_3 = 2u) {

    }
  }
  s.erase(end + 1);
}

inline bool next_pos_is(const std::string_view source, const size_t pos,
                        const std::initializer_list<char> chars,
                        const size_t n = 1) noexcept {
  const size_t idx = pos + n;
    {
    const size_t emel_branch_4 = static_cast<size_t>(idx >= source.size());
    for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 1u; emel_case_4 = 2u) {
            return false;
    }
    for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 0u; emel_case_4 = 2u) {

    }
  }
  for (const char c : chars) {
        {
      const size_t emel_branch_5 = static_cast<size_t>(source[idx] == c);
      for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 1u; emel_case_5 = 2u) {
                return true;
      }
      for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 0u; emel_case_5 = 2u) {

      }
    }
  }
  return false;
}

inline bool decode_escape(const char ch, char &out) noexcept {
  const size_t is_n = static_cast<size_t>(ch == 'n');
  const size_t is_t = static_cast<size_t>(ch == 't');
  const size_t is_r = static_cast<size_t>(ch == 'r');
  const size_t is_b = static_cast<size_t>(ch == 'b');
  const size_t is_f = static_cast<size_t>(ch == 'f');
  const size_t is_v = static_cast<size_t>(ch == 'v');
  const size_t is_backslash = static_cast<size_t>(ch == '\\');
  const size_t is_single_quote = static_cast<size_t>(ch == '\'');
  const size_t is_double_quote = static_cast<size_t>(ch == '"');
  const size_t code = is_n * 1u + is_t * 2u + is_r * 3u + is_b * 4u + is_f * 5u +
                      is_v * 6u + is_backslash * 7u + is_single_quote * 8u +
                      is_double_quote * 9u;
  constexpr std::array<char, 10> decoded = {
      '\0',
      '\n',
      '\t',
      '\r',
      '\b',
      '\f',
      '\v',
      '\\',
      '\'',
      '"',
  };
  out = decoded[code];
  return code != 0u;
}

inline bool is_closing_block(const std::string_view source,
                             const size_t pos) noexcept {
  return pos < source.size() && source[pos] == '-' &&
         next_pos_is(source, pos, {'%', '}'});
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
  {
    const size_t emel_branch_6 = static_cast<size_t>(closes_block);
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 1u; emel_case_6 = 2u) {
      next.last_block_can_trim_newline = true;
      next.last_block_rstrip = token_text.size() >= 3 && token_text[0] == '-' &&
                               token_text.back() == '}';
    }
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 0u; emel_case_6 = 2u) {

    }
  }
  {
    const size_t emel_branch_7 = static_cast<size_t>(type == token_type::comment);
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 1u; emel_case_7 = 2u) {
      next.last_block_can_trim_newline = true;
    }
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 0u; emel_case_7 = 2u) {

    }
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
    const size_t emel_branch_literal = static_cast<size_t>(source[pos] != '\\');
    for (size_t emel_case_literal = emel_branch_literal; emel_case_literal == 1u;
         emel_case_literal = 2u) {
      value.push_back(source[pos]);
      ++pos;
    }
    for (size_t emel_case_literal = emel_branch_literal; emel_case_literal == 0u;
         emel_case_literal = 2u) {
      ++pos;
          {
        const size_t emel_branch_8 = static_cast<size_t>(pos >= source.size());
        for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 1u; emel_case_8 = 2u) {
                    set_error(out, pos);
                    return value;
        }
        for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 0u; emel_case_8 = 2u) {

        }
      }

      char decoded = '\0';
      const char escaped = source[pos];
          {
        const size_t emel_branch_9 = static_cast<size_t>(!decode_escape(escaped, decoded));
        for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 1u; emel_case_9 = 2u) {
                    set_error(out, pos);
                    return value;
        }
        for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 0u; emel_case_9 = 2u) {

        }
      }
      value.push_back(decoded);
      ++pos;
    }
  }
  return value;
}

inline std::string consume_numeric(const std::string_view source, size_t &pos) {
  std::string value;
  while (pos < source.size() && is_integer(source[pos])) {
    value.push_back(source[pos]);
    ++pos;
  }
  const bool has_fraction = pos < source.size() && source[pos] == '.' &&
      pos + 1 < source.size() && is_integer(source[pos + 1]);
    {
    const size_t emel_branch_10 = static_cast<size_t>(has_fraction);
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 1u; emel_case_10 = 2u) {
            value.push_back(source[pos]);
            ++pos;
            while (pos < source.size() && is_integer(source[pos])) {
              value.push_back(source[pos]);
              ++pos;
            }
    }
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 0u; emel_case_10 = 2u) {

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
    {
      const size_t emel_branch_text_boundary =
          static_cast<size_t>(at_text_boundary(cursor.last_token_type));
      for (size_t emel_case_text_boundary = emel_branch_text_boundary;
           emel_case_text_boundary == 1u;
           emel_case_text_boundary = 2u) {
        const size_t start = pos;
        size_t end = start;
        while (pos < size && !(source[pos] == '{' &&
                               next_pos_is(source, pos, {'%', '{', '#'}))) {
          end = ++pos;
        }

        const bool has_opening_block = pos < size && source[pos] == '{' &&
            next_pos_is(source, pos, {'%', '#', '-'});
        {
          const size_t emel_branch_opening = static_cast<size_t>(has_opening_block);
          for (size_t emel_case_opening = emel_branch_opening; emel_case_opening == 1u;
               emel_case_opening = 2u) {
            size_t current = end;
            bool keep_trimming = true;
            while (current > start && keep_trimming) {
              const char c = source[current - 1];
              const size_t trim_mode =
                  static_cast<size_t>(current == 1) * 2u + static_cast<size_t>(c == '\n');
              using trim_handler_t =
                  void (*)(size_t &, size_t &, bool &, const char) noexcept;
              static constexpr std::array<trim_handler_t, 3> trim_handlers = {
                  +[](size_t &, size_t & current_value, bool & keep_value,
                      const char c_value) noexcept {
                    const size_t emel_branch_is_space = static_cast<size_t>(is_space(c_value));
                    for (size_t emel_case_is_space = emel_branch_is_space;
                         emel_case_is_space == 1u;
                         emel_case_is_space = 2u) {
                      --current_value;
                    }
                    for (size_t emel_case_is_space = emel_branch_is_space;
                         emel_case_is_space == 0u;
                         emel_case_is_space = 2u) {
                      keep_value = false;
                    }
                  },
                  +[](size_t & end_value, size_t & current_value, bool & keep_value,
                      const char) noexcept {
                    end_value = current_value;
                    keep_value = false;
                  },
                  +[](size_t & end_value, size_t &, bool & keep_value, const char) noexcept {
                    end_value = 0;
                    keep_value = false;
                  },
              };
              static constexpr std::array<size_t, 4> trim_mode_dispatch = {0u, 1u, 2u, 0u};
              trim_handlers[trim_mode_dispatch[trim_mode]](end, current, keep_trimming, c);
            }
          }
          for (size_t emel_case_opening = emel_branch_opening; emel_case_opening == 0u;
               emel_case_opening = 2u) {

          }
        }

        std::string text = std::string(source.substr(start, end - start));
        const bool trim_leading_newline =
            cursor.last_block_can_trim_newline && !text.empty() && text.front() == '\n';
        {
          const size_t emel_branch_trim_leading_newline =
              static_cast<size_t>(trim_leading_newline);
          for (size_t emel_case_trim_leading_newline = emel_branch_trim_leading_newline;
               emel_case_trim_leading_newline == 1u;
               emel_case_trim_leading_newline = 2u) {
            text.erase(text.begin());
          }
          for (size_t emel_case_trim_leading_newline = emel_branch_trim_leading_newline;
               emel_case_trim_leading_newline == 0u;
               emel_case_trim_leading_newline = 2u) {

          }
        }
        {
          const size_t emel_branch_lstrip = static_cast<size_t>(cursor.last_block_rstrip);
          for (size_t emel_case_lstrip = emel_branch_lstrip; emel_case_lstrip == 1u;
               emel_case_lstrip = 2u) {
            string_lstrip(text, " \t\r\n");
          }
          for (size_t emel_case_lstrip = emel_branch_lstrip; emel_case_lstrip == 0u;
               emel_case_lstrip = 2u) {

          }
        }

        const bool is_lstrip_block = pos < size && source[pos] == '{' &&
                                     next_pos_is(source, pos, {'{', '%', '#'}) &&
                                     next_pos_is(source, pos, {'-'}, 2);
        {
          const size_t emel_branch_rstrip = static_cast<size_t>(is_lstrip_block);
          for (size_t emel_case_rstrip = emel_branch_rstrip; emel_case_rstrip == 1u;
               emel_case_rstrip = 2u) {
            string_rstrip(text, " \t\r\n");
          }
          for (size_t emel_case_rstrip = emel_branch_rstrip; emel_case_rstrip == 0u;
               emel_case_rstrip = 2u) {

          }
        }

        {
          const size_t emel_branch_has_text = static_cast<size_t>(!text.empty());
          for (size_t emel_case_has_text = emel_branch_has_text; emel_case_has_text == 1u;
               emel_case_has_text = 2u) {
            out.has_token = true;
            out.token_value = token{token_type::text, std::move(text), start};
            out.next_cursor = emit_cursor(cursor, pos, out.token_value.type,
                                          out.token_value.value);
            return out;
          }
          for (size_t emel_case_has_text = emel_branch_has_text; emel_case_has_text == 0u;
               emel_case_has_text = 2u) {

          }
        }
      }
      for (size_t emel_case_text_boundary = emel_branch_text_boundary;
           emel_case_text_boundary == 0u;
           emel_case_text_boundary = 2u) {

      }
    }

    {
      const size_t emel_branch_comment =
          static_cast<size_t>(source[pos] == '{' && next_pos_is(source, pos, {'#'}));
      for (size_t emel_case_comment = emel_branch_comment; emel_case_comment == 1u;
           emel_case_comment = 2u) {
        const size_t start = pos;
        pos += 2;
        std::string comment;
        while (pos < size &&
               !(source[pos] == '#' && next_pos_is(source, pos, {'}'}))) {
          {
            const size_t emel_branch_11 = static_cast<size_t>(pos + 2 >= size);
            for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 1u;
                 emel_case_11 = 2u) {
              set_error(out, pos);
              return out;
            }
            for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 0u;
                 emel_case_11 = 2u) {

            }
          }
          comment.push_back(source[pos]);
          ++pos;
        }
        {
          const size_t emel_branch_12 = static_cast<size_t>(pos + 1 >= size);
          for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 1u;
               emel_case_12 = 2u) {
            set_error(out, pos);
            return out;
          }
          for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 0u;
               emel_case_12 = 2u) {

          }
        }
        pos += 2;
        out.has_token = true;
        out.token_value = token{token_type::comment, std::move(comment), start};
        out.next_cursor =
            emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
        return out;
      }
      for (size_t emel_case_comment = emel_branch_comment; emel_case_comment == 0u;
           emel_case_comment = 2u) {

      }
    }

    const bool starts_trim =
        source[pos] == '-' &&
        (cursor.last_token_type == token_type::open_expression ||
         cursor.last_token_type == token_type::open_statement);
    {
      const size_t emel_branch_starts_trim = static_cast<size_t>(starts_trim);
      for (size_t emel_case_starts_trim = emel_branch_starts_trim;
           emel_case_starts_trim == 1u;
           emel_case_starts_trim = 2u) {
        ++pos;
        {
          const size_t emel_branch_13 = static_cast<size_t>(pos >= size);
          for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 1u;
               emel_case_13 = 2u) {
            out.next_cursor = cursor;
            out.next_cursor.offset = pos;
            return out;
          }
          for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 0u;
               emel_case_13 = 2u) {

          }
        }
      }
      for (size_t emel_case_starts_trim = emel_branch_starts_trim;
           emel_case_starts_trim == 0u;
           emel_case_starts_trim = 2u) {

      }
    }

    while (pos < size && is_space(source[pos])) {
      ++pos;
    }
    {
      const size_t emel_branch_14 = static_cast<size_t>(pos >= size);
      for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 1u;
           emel_case_14 = 2u) {
        out.next_cursor = cursor;
        out.next_cursor.offset = pos;
        return out;
      }
      for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 0u;
           emel_case_14 = 2u) {

      }
    }

    const char ch = source[pos];
    const bool unary_or_sign = !is_closing_block(source, pos) && (ch == '-' || ch == '+');
    {
      const size_t emel_branch_unary_or_sign = static_cast<size_t>(unary_or_sign);
      for (size_t emel_case_unary_or_sign = emel_branch_unary_or_sign;
           emel_case_unary_or_sign == 1u;
           emel_case_unary_or_sign = 2u) {
        const bool invalid_prefix_context =
            cursor.last_token_type == token_type::text ||
            cursor.last_token_type == token_type::eof;
        {
          const size_t emel_branch_invalid_prefix =
              static_cast<size_t>(invalid_prefix_context);
          for (size_t emel_case_invalid_prefix = emel_branch_invalid_prefix;
               emel_case_invalid_prefix == 1u;
               emel_case_invalid_prefix = 2u) {
            set_error(out, pos);
            return out;
          }
          for (size_t emel_case_invalid_prefix = emel_branch_invalid_prefix;
               emel_case_invalid_prefix == 0u;
               emel_case_invalid_prefix = 2u) {

          }
        }
        {
          const size_t emel_branch_allowed =
              static_cast<size_t>(unary_prefix_allowed(cursor.last_token_type));
          for (size_t emel_case_allowed = emel_branch_allowed; emel_case_allowed == 1u;
               emel_case_allowed = 2u) {
            const size_t start = pos;
            ++pos;
            std::string num = consume_numeric(source, pos);
            std::string value;
            value.reserve(num.size() + 1);
            value.push_back(ch);
            value += num;
            constexpr std::array<token_type, 2> type_candidates = {
                token_type::numeric_literal,
                token_type::unary_operator,
            };
            const token_type type = type_candidates[static_cast<size_t>(num.empty())];
            out.has_token = true;
            out.token_value = token{type, std::move(value), start};
            out.next_cursor = emit_cursor(cursor, pos, out.token_value.type,
                                          out.token_value.value);
            return out;
          }
          for (size_t emel_case_allowed = emel_branch_allowed; emel_case_allowed == 0u;
               emel_case_allowed = 2u) {

          }
        }
      }
      for (size_t emel_case_unary_or_sign = emel_branch_unary_or_sign;
           emel_case_unary_or_sign == 0u;
           emel_case_unary_or_sign = 2u) {

      }
    }

    for (const auto &entry : k_mapping_table) {
      const bool skip_close_curly = entry.seq == "}}" && cursor.curly_bracket_depth > 0;
      {
        const size_t emel_branch_eval_match = static_cast<size_t>(!skip_close_curly);
        for (size_t emel_case_eval_match = emel_branch_eval_match; emel_case_eval_match == 1u;
             emel_case_eval_match = 2u) {
          const bool match = pos + entry.seq.size() <= size &&
              source.compare(pos, entry.seq.size(), entry.seq) == 0;
          {
            const size_t emel_branch_match = static_cast<size_t>(match);
            for (size_t emel_case_match = emel_branch_match; emel_case_match == 1u;
                 emel_case_match = 2u) {
              out.has_token = true;
              out.token_value = token{entry.type, std::string(entry.seq), pos};
              out.next_cursor =
                  emit_cursor(cursor, pos + entry.seq.size(), out.token_value.type,
                              out.token_value.value);
              return out;
            }
            for (size_t emel_case_match = emel_branch_match; emel_case_match == 0u;
                 emel_case_match = 2u) {

            }
          }
        }
        for (size_t emel_case_eval_match = emel_branch_eval_match; emel_case_eval_match == 0u;
             emel_case_eval_match = 2u) {

        }
      }
    }

    {
      const size_t emel_branch_quote = static_cast<size_t>(ch == '\'' || ch == '"');
      for (size_t emel_case_quote = emel_branch_quote; emel_case_quote == 1u;
           emel_case_quote = 2u) {
        const size_t start = pos;
        ++pos;
        std::string value = consume_escaped_until(source, pos, ch, out);
        {
          const size_t emel_branch_err =
              static_cast<size_t>(out.err != error_code(parser::error::none));
          for (size_t emel_case_err = emel_branch_err; emel_case_err == 1u;
               emel_case_err = 2u) {
            return out;
          }
          for (size_t emel_case_err = emel_branch_err; emel_case_err == 0u;
               emel_case_err = 2u) {

          }
        }
        {
          const size_t emel_branch_pos = static_cast<size_t>(pos >= size);
          for (size_t emel_case_pos = emel_branch_pos; emel_case_pos == 1u;
               emel_case_pos = 2u) {
            set_error(out, pos);
            return out;
          }
          for (size_t emel_case_pos = emel_branch_pos; emel_case_pos == 0u;
               emel_case_pos = 2u) {

          }
        }
        ++pos;
        out.has_token = true;
        out.token_value =
            token{token_type::string_literal, std::move(value), start};
        out.next_cursor =
            emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
        return out;
      }
      for (size_t emel_case_quote = emel_branch_quote; emel_case_quote == 0u;
           emel_case_quote = 2u) {

      }
    }

    {
      const size_t emel_branch_integer = static_cast<size_t>(is_integer(ch));
      for (size_t emel_case_integer = emel_branch_integer; emel_case_integer == 1u;
           emel_case_integer = 2u) {
        const size_t start = pos;
        std::string value = consume_numeric(source, pos);
        out.has_token = true;
        out.token_value =
            token{token_type::numeric_literal, std::move(value), start};
        out.next_cursor =
            emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
        return out;
      }
      for (size_t emel_case_integer = emel_branch_integer; emel_case_integer == 0u;
           emel_case_integer = 2u) {

      }
    }

    {
      const size_t emel_branch_word = static_cast<size_t>(is_word(ch));
      for (size_t emel_case_word = emel_branch_word; emel_case_word == 1u;
           emel_case_word = 2u) {
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
      for (size_t emel_case_word = emel_branch_word; emel_case_word == 0u;
           emel_case_word = 2u) {

      }
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
  using scan_fn_t = scan_outcome (*)(const ::emel::text::jinja::lexer::cursor &);
  static constexpr std::array<scan_fn_t, 2> scan_fns = {
      +[](const ::emel::text::jinja::lexer::cursor & value) -> scan_outcome {
        return scan_next_token(value);
      },
      +[](const ::emel::text::jinja::lexer::cursor &) -> scan_outcome {
        return scan_outcome{};
      },
  };
  return scan_fns[static_cast<size_t>(invalid_source || invalid_offset)](cursor);
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
    {
      const size_t emel_branch_terminal = static_cast<size_t>(terminal);
      for (size_t emel_case_terminal = emel_branch_terminal; emel_case_terminal == 1u;
           emel_case_terminal = 2u) {
        return plan;
      }
      for (size_t emel_case_terminal = emel_branch_terminal; emel_case_terminal == 0u;
           emel_case_terminal = 2u) {

      }
    }
    cursor = scan.next_cursor;
  }
}

} // namespace emel::text::jinja::lexer::detail
