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
  const size_t erase_count = select_size(start == std::string::npos, s.size(), start);
  s.erase(0, erase_count);
}

inline void string_rstrip(std::string &s, const char *chars) {
  const size_t end = s.find_last_not_of(chars);
  const size_t keep_count = select_size(end == std::string::npos, 0u, end + 1u);
  s.erase(keep_count);
}

inline bool next_pos_is(const std::string_view source, const size_t pos,
                        const std::initializer_list<char> chars,
                        const size_t n = 1) noexcept {
  const size_t idx = pos + n;
  const bool in_range = idx < source.size();
  const char candidate = view_char_at_or(source, idx, '\0');
  bool matched = false;
  for (const char c : chars) {
    matched = matched || candidate == c;
  }
  return in_range && matched;
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

inline void append_char_none(std::string &, const char) noexcept {}

inline void append_char_some(std::string &value, const char ch) noexcept {
  value.push_back(ch);
}

inline void set_error_none(scan_outcome &, const size_t, bool &) noexcept {}

inline void set_error_some(scan_outcome &out, const size_t pos, bool &ok) noexcept {
  set_error(out, pos);
  ok = false;
}

inline void set_error_done_none(scan_outcome &, const size_t, bool &) noexcept {}

inline void set_error_done_some(scan_outcome &out, const size_t pos, bool &done) noexcept {
  set_error(out, pos);
  done = true;
}

inline void emit_no_token_cursor(scan_outcome &out,
                                 const ::emel::text::jinja::lexer::cursor &cursor,
                                 const size_t pos) noexcept {
  out.has_token = false;
  out.next_cursor = cursor;
  out.next_cursor.offset = pos;
}

inline void consume_escape_literal(const std::string_view source,
                                   size_t &pos,
                                   std::string &value,
                                   scan_outcome &,
                                   bool &) noexcept {
  value.push_back(source[pos]);
  ++pos;
}

inline void consume_escape_sequence(const std::string_view source,
                                    size_t &pos,
                                    std::string &value,
                                    scan_outcome &out,
                                    bool &ok) noexcept {
  ++pos;
  const bool in_range = pos < source.size();
  const char escaped = view_char_at_or(source, pos, '\0');
  char decoded = '\0';
  const bool decode_ok = in_range && decode_escape(escaped, decoded);

  using error_handler_t = void (*)(scan_outcome &, size_t, bool &) noexcept;
  const error_handler_t error_handlers[2] = {
      set_error_none,
      set_error_some,
  };
  error_handlers[static_cast<size_t>(!decode_ok)](out, pos, ok);

  using append_handler_t = void (*)(std::string &, char) noexcept;
  const append_handler_t append_handlers[2] = {
      append_char_none,
      append_char_some,
  };
  append_handlers[static_cast<size_t>(decode_ok)](value, decoded);
  pos += static_cast<size_t>(decode_ok);
}

inline std::string consume_escaped_until(const std::string_view source,
                                         size_t &pos, const char terminal,
                                         scan_outcome &out) {
  std::string value;
  bool ok = true;
  using consume_handler_t =
      void (*)(std::string_view, size_t &, std::string &, scan_outcome &, bool &) noexcept;
  const consume_handler_t consume_handlers[2] = {
      consume_escape_sequence,
      consume_escape_literal,
  };
  while (ok && pos < source.size() && source[pos] != terminal) {
    const bool literal = source[pos] != '\\';
    consume_handlers[static_cast<size_t>(literal)](source, pos, value, out, ok);
  }
  return value;
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

inline void trim_text_before_opening_block(const std::string_view source,
                                           const size_t start,
                                           size_t &end,
                                           const size_t pos,
                                           const size_t size) noexcept {
  const bool has_opening_block =
      pos < size &&
      source[pos] == '{' &&
      next_pos_is(source, pos, {'%', '#', '-'});
  using trim_entry_t =
      void (*)(const std::string_view, const size_t, size_t &, const size_t, const size_t)
          noexcept;
  static constexpr std::array<trim_entry_t, 2> trim_entries = {
      +[](const std::string_view, const size_t, size_t &, const size_t, const size_t) noexcept {},
      +[](const std::string_view source_value,
          const size_t start_value,
          size_t &end_value,
          const size_t,
          const size_t) noexcept {
        size_t current = end_value;
        bool keep_trimming = true;
        while (current > start_value && keep_trimming) {
          const char c = source_value[current - 1];
          const size_t trim_mode =
              static_cast<size_t>(current == 1u) * 2u +
              static_cast<size_t>(c == '\n');
          using trim_handler_t = void (*)(size_t &, size_t &, bool &, const char) noexcept;
          static constexpr std::array<trim_handler_t, 3> trim_handlers = {
              +[](size_t &, size_t &current_value, bool &keep_value, const char c_value) noexcept {
                if (is_space(c_value)) {
                  --current_value;
                } else {
                  keep_value = false;
                }
              },
              +[](size_t &end_cursor, size_t &current_value, bool &keep_value, const char)
                  noexcept {
                end_cursor = current_value;
                keep_value = false;
              },
              +[](size_t &end_cursor, size_t &, bool &keep_value, const char) noexcept {
                end_cursor = 0u;
                keep_value = false;
              },
          };
          static constexpr std::array<size_t, 4> trim_mode_dispatch = {0u, 1u, 2u, 0u};
          trim_handlers[trim_mode_dispatch[trim_mode]](end_value, current, keep_trimming, c);
        }
      },
  };
  trim_entries[static_cast<size_t>(has_opening_block)](source, start, end, pos, size);
}

inline bool scan_text_token_at_boundary(const ::emel::text::jinja::lexer::cursor &cursor,
                                        scan_outcome &out,
                                        const std::string_view source,
                                        const size_t size,
                                        size_t &pos) {
  const bool at_boundary = at_text_boundary(cursor.last_token_type);
  using scan_entry_t = bool (*)(const ::emel::text::jinja::lexer::cursor &,
                                scan_outcome &,
                                const std::string_view,
                                const size_t,
                                size_t &);
  static constexpr std::array<scan_entry_t, 2> scan_entries = {
      +[](const ::emel::text::jinja::lexer::cursor &,
          scan_outcome &,
          const std::string_view,
          const size_t,
          size_t &) -> bool {
        return false;
      },
      +[](const ::emel::text::jinja::lexer::cursor &cursor_value,
          scan_outcome &out_value,
          const std::string_view source_value,
          const size_t size_value,
          size_t &pos_value) -> bool {
        const size_t start = pos_value;
        size_t end = start;
        while (pos_value < size_value &&
               !(source_value[pos_value] == '{' &&
                 next_pos_is(source_value, pos_value, {'%', '{', '#'}))) {
          end = ++pos_value;
        }

        trim_text_before_opening_block(source_value, start, end, pos_value, size_value);

        std::string text = std::string(source_value.substr(start, end - start));
        const bool trim_leading_newline =
            cursor_value.last_block_can_trim_newline && !text.empty() && text.front() == '\n';
        if (trim_leading_newline) {
          text.erase(text.begin());
        }
        if (cursor_value.last_block_rstrip) {
          string_lstrip(text, " \t\r\n");
        }

        const bool is_lstrip_block =
            pos_value < size_value &&
            source_value[pos_value] == '{' &&
            next_pos_is(source_value, pos_value, {'{', '%', '#'}) &&
            next_pos_is(source_value, pos_value, {'-'}, 2u);
        if (is_lstrip_block) {
          string_rstrip(text, " \t\r\n");
        }

        if (text.empty()) {
          return false;
        }
        out_value.has_token = true;
        out_value.token_value = token{token_type::text, std::move(text), start};
        out_value.next_cursor = emit_cursor(
            cursor_value, pos_value, out_value.token_value.type, out_value.token_value.value);
        return true;
      },
  };
  return scan_entries[static_cast<size_t>(at_boundary)](cursor, out, source, size, pos);
}

inline bool scan_comment_token(const ::emel::text::jinja::lexer::cursor &cursor,
                               scan_outcome &out,
                               const std::string_view source,
                               const size_t size,
                               size_t &pos) {
  const bool starts_comment = pos < size && source[pos] == '{' && next_pos_is(source, pos, {'#'});
  using comment_entry_t = bool (*)(const ::emel::text::jinja::lexer::cursor &,
                                   scan_outcome &,
                                   const std::string_view,
                                   const size_t,
                                   size_t &);
  static constexpr std::array<comment_entry_t, 2> comment_entries = {
      +[](const ::emel::text::jinja::lexer::cursor &,
          scan_outcome &,
          const std::string_view,
          const size_t,
          size_t &) -> bool {
        return false;
      },
      +[](const ::emel::text::jinja::lexer::cursor &cursor_value,
          scan_outcome &out_value,
          const std::string_view source_value,
          const size_t size_value,
          size_t &pos_value) -> bool {
        const size_t start = pos_value;
        pos_value += 2u;
        std::string comment;
        bool done = false;
        while (pos_value < size_value &&
               !(source_value[pos_value] == '#' && next_pos_is(source_value, pos_value, {'}'})) &&
               !done) {
          const bool hit_unterminated = pos_value + 2u >= size_value;
          using unterminated_entry_t = void (*)(scan_outcome &, const size_t, bool &) noexcept;
          static constexpr std::array<unterminated_entry_t, 2> unterminated_entries = {
              +[](scan_outcome &, const size_t, bool &) noexcept {},
              +[](scan_outcome &out_entry, const size_t value_pos, bool &done_entry) noexcept {
                set_error(out_entry, value_pos);
                done_entry = true;
              },
          };
          unterminated_entries[static_cast<size_t>(hit_unterminated)](out_value, pos_value, done);

          if (!done) {
            comment.push_back(source_value[pos_value]);
            ++pos_value;
          }
        }

        if (done) {
          return true;
        }

        const bool missing_close = pos_value + 1u >= size_value;
        using close_entry_t = bool (*)(scan_outcome &, const size_t);
        static constexpr std::array<close_entry_t, 2> close_entries = {
            +[](scan_outcome &, const size_t) -> bool { return false; },
            +[](scan_outcome &out_entry, const size_t value_pos) -> bool {
              set_error(out_entry, value_pos);
              return true;
            },
        };
        if (close_entries[static_cast<size_t>(missing_close)](out_value, pos_value)) {
          return true;
        }

        pos_value += 2u;
        out_value.has_token = true;
        out_value.token_value = token{token_type::comment, std::move(comment), start};
        out_value.next_cursor = emit_cursor(
            cursor_value, pos_value, out_value.token_value.type, out_value.token_value.value);
        return true;
      },
  };
  return comment_entries[static_cast<size_t>(starts_comment)](cursor, out, source, size, pos);
}

inline scan_outcome
scan_next_token(const ::emel::text::jinja::lexer::cursor &cursor) {
  scan_outcome out{};
  out.next_cursor = cursor;

  const std::string_view source = cursor.source;
  const size_t size = source.size();
  size_t pos = cursor.offset;

  while (pos < size) {
    if (scan_text_token_at_boundary(cursor, out, source, size, pos)) {
      return out;
    }

    if (scan_comment_token(cursor, out, source, size, pos)) {
      return out;
    }

    const bool starts_trim =
        pos < size &&
        source[pos] == '-' &&
        (cursor.last_token_type == token_type::open_expression ||
         cursor.last_token_type == token_type::open_statement);
    if (starts_trim) {
      ++pos;
      if (pos >= size) {
        emit_no_token_cursor(out, cursor, pos);
        return out;
      }
    }

    while (pos < size && is_space(source[pos])) {
      ++pos;
    }
    if (pos >= size) {
      emit_no_token_cursor(out, cursor, pos);
      return out;
    }

    const char ch = source[pos];
    const bool unary_or_sign = !is_closing_block(source, pos) && (ch == '-' || ch == '+');
    if (unary_or_sign) {
      const bool invalid_prefix_context =
          cursor.last_token_type == token_type::text ||
          cursor.last_token_type == token_type::eof;
      if (invalid_prefix_context) {
        set_error(out, pos);
        return out;
      }
      if (unary_prefix_allowed(cursor.last_token_type)) {
        const size_t start = pos;
        ++pos;
        std::string num = consume_numeric(source, pos);
        std::string value;
        value.reserve(num.size() + 1u);
        value.push_back(ch);
        value += num;
        constexpr std::array<token_type, 2> type_candidates = {
            token_type::numeric_literal,
            token_type::unary_operator,
        };
        const token_type type = type_candidates[static_cast<size_t>(num.empty())];
        out.has_token = true;
        out.token_value = token{type, std::move(value), start};
        out.next_cursor = emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
        return out;
      }
    }

    for (const auto &entry : k_mapping_table) {
      const bool skip_close_curly = entry.seq == "}}" && cursor.curly_bracket_depth > 0u;
      if (!skip_close_curly &&
          pos + entry.seq.size() <= size &&
          source.compare(pos, entry.seq.size(), entry.seq) == 0) {
        out.has_token = true;
        out.token_value = token{entry.type, std::string(entry.seq), pos};
        out.next_cursor = emit_cursor(cursor,
                                      pos + entry.seq.size(),
                                      out.token_value.type,
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
      out.token_value = token{token_type::string_literal, std::move(value), start};
      out.next_cursor = emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
      return out;
    }

    if (is_integer(ch)) {
      const size_t start = pos;
      std::string value = consume_numeric(source, pos);
      out.has_token = true;
      out.token_value = token{token_type::numeric_literal, std::move(value), start};
      out.next_cursor = emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
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
      out.next_cursor = emit_cursor(cursor, pos, out.token_value.type, out.token_value.value);
      return out;
    }

    set_error(out, pos);
    return out;
  }

  emit_no_token_cursor(out, cursor, pos);
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

  bool terminal = false;
  while (!terminal) {
    const scan_outcome scan = scan_next_token_safe(cursor);
    plan.outcomes.push_back(scan);
    terminal = scan.err != error_code(parser::error::none) || !scan.has_token;
    cursor = scan.next_cursor;
  }
  return plan;
}

} // namespace emel::text::jinja::lexer::detail
