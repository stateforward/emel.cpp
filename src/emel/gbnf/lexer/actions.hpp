#pragma once

#include <cstdint>
#include <string_view>

#include "emel/gbnf/lexer/context.hpp"
#include "emel/gbnf/lexer/errors.hpp"
#include "emel/gbnf/lexer/events.hpp"

namespace emel::gbnf::lexer::action {

inline constexpr int32_t error_code(const emel::gbnf::lexer::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

namespace detail {

inline bool is_word_char(const char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' ||
         (c >= '0' && c <= '9');
}

inline bool is_newline_char(const char c) noexcept {
  return c == '\n' || c == '\r';
}

inline uint32_t skip_layout(std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  while (pos < size) {
    const char c = input[pos];
    if (c == ' ' || c == '\t') {
      ++pos;
      continue;
    }
    if (c == '#') {
      ++pos;
      while (pos < size && !is_newline_char(input[pos])) {
        ++pos;
      }
      continue;
    }
    break;
  }
  return pos;
}

inline bool has_prefix(std::string_view input, uint32_t pos, std::string_view prefix) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  if (pos + prefix.size() > size) {
    return false;
  }
  return input.substr(pos, prefix.size()) == prefix;
}

inline uint32_t scan_quoted(std::string_view input, uint32_t pos, const char terminator) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  ++pos;  // opening quote/bracket already consumed by caller.
  while (pos < size) {
    const char c = input[pos];
    if (c == '\\' && pos + 1u < size) {
      pos += 2u;
      continue;
    }
    ++pos;
    if (c == terminator) {
      break;
    }
  }
  return pos;
}

inline uint32_t scan_braced_quantifier(std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  ++pos;  // consume '{'
  while (pos < size && input[pos] != '}') {
    ++pos;
  }
  if (pos < size && input[pos] == '}') {
    ++pos;
  }
  return pos;
}

inline uint32_t scan_token_ref(std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  if (input[pos] == '!') {
    ++pos;
  }
  if (pos + 1u >= size || input[pos] != '<' || input[pos + 1u] != '[') {
    return pos;
  }
  pos += 2u;
  while (pos < size && input[pos] >= '0' && input[pos] <= '9') {
    ++pos;
  }
  if (pos + 1u < size && input[pos] == ']' && input[pos + 1u] == '>') {
    return pos + 2u;
  }
  return pos;
}

inline event::token make_token(const std::string_view input,
                               const uint32_t start,
                               const uint32_t end,
                               const event::token_kind kind) noexcept {
  return event::token{
      .kind = kind,
      .text = input.substr(start, end - start),
      .start = start,
      .end = end,
  };
}

inline event::token scan_token(const lexer::cursor &cursor, uint32_t &next_offset) noexcept {
  const std::string_view input = cursor.input;
  const uint32_t size = static_cast<uint32_t>(input.size());
  uint32_t pos = skip_layout(input, cursor.offset);

  if (pos >= size) {
    next_offset = pos;
    return {};
  }

  const uint32_t start = pos;
  const char c = input[pos];

  if (is_newline_char(c)) {
    if (c == '\r' && pos + 1u < size && input[pos + 1u] == '\n') {
      pos += 2u;
    } else {
      ++pos;
    }
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::newline);
  }

  if (has_prefix(input, pos, "::=")) {
    pos += 3u;
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::definition_operator);
  }

  if (c == '|') {
    ++pos;
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::alternation);
  }

  if (c == '.') {
    ++pos;
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::dot);
  }

  if (c == '(') {
    ++pos;
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::open_group);
  }

  if (c == ')') {
    ++pos;
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::close_group);
  }

  if (c == '+' || c == '*' || c == '?') {
    ++pos;
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::quantifier);
  }

  if (c == '"') {
    pos = scan_quoted(input, pos, '"');
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::string_literal);
  }

  if (c == '[') {
    pos = scan_quoted(input, pos, ']');
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::character_class);
  }

  if (c == '<' || (c == '!' && has_prefix(input, pos + 1u, "<["))) {
    const uint32_t end = scan_token_ref(input, pos);
    if (end > pos) {
      next_offset = end;
      return make_token(input, start, end, event::token_kind::rule_reference);
    }
  }

  if (is_word_char(c)) {
    ++pos;
    while (pos < size && is_word_char(input[pos])) {
      ++pos;
    }
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::identifier);
  }

  if (c == '{') {
    pos = scan_braced_quantifier(input, pos);
    next_offset = pos;
    return make_token(input, start, pos, event::token_kind::quantifier);
  }

  ++pos;
  next_offset = pos;
  return make_token(input, start, pos, event::token_kind::unknown);
}

}  // namespace detail

struct emit_next_token {
  void operator()(const event::next &ev, context &) const noexcept {
    uint32_t next_offset = ev.cursor.offset;
    const event::token token = detail::scan_token(ev.cursor, next_offset);
    lexer::cursor next_cursor = ev.cursor;
    next_cursor.offset = next_offset;
    next_cursor.token_count += 1;
    ev.on_done(events::next_done{
        .token = token,
        .has_token = true,
        .next_cursor = next_cursor,
    });
  }
};

struct emit_eof {
  void operator()(const event::next &ev, context &) const noexcept {
    ev.on_done(events::next_done{
        .token = {},
        .has_token = false,
        .next_cursor = ev.cursor,
    });
  }
};

struct reject_invalid_next {
  void operator()(const event::next &ev, context &) const noexcept {
    ev.on_error(events::next_error{error_code(error::invalid_request)});
  }
};

struct reject_invalid_cursor {
  void operator()(const event::next &ev, context &) const noexcept {
    ev.on_error(events::next_error{error_code(error::invalid_request)});
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.on_error; }) {
      if (ev.on_error) {
        ev.on_error(events::next_error{error_code(error::internal_error)});
      }
    }
  }
};

inline constexpr emit_next_token emit_next_token{};
inline constexpr emit_eof emit_eof{};
inline constexpr reject_invalid_next reject_invalid_next{};
inline constexpr reject_invalid_cursor reject_invalid_cursor{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::lexer::action
