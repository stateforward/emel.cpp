#pragma once

#include <cstdint>
#include <string_view>

#include "emel/gbnf/rule_parser/lexer/context.hpp"
#include "emel/gbnf/rule_parser/lexer/errors.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"

namespace emel::gbnf::rule_parser::lexer::action {

inline constexpr int32_t error_code(const emel::gbnf::rule_parser::lexer::error err) noexcept {
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
  uint32_t scan_more = 1;
  while (pos < size && scan_more != 0u) {
    const char c = input[pos];
    const size_t mode =
        static_cast<size_t>(c == ' ' || c == '\t') +
        (static_cast<size_t>(c == '#') * 2u);
    const size_t advance_space = static_cast<size_t>(mode == 1u);
    const size_t skip_comment = static_cast<size_t>(mode == 2u);
    pos += static_cast<uint32_t>(advance_space);
    {
      const size_t emel_branch_skip_comment = skip_comment;
      for (size_t emel_case_skip_comment = emel_branch_skip_comment;
           emel_case_skip_comment == 1u;
           emel_case_skip_comment = 2u) {
        ++pos;
        while (pos < size && !is_newline_char(input[pos])) {
          ++pos;
        }
      }
      for (size_t emel_case_skip_comment = emel_branch_skip_comment;
           emel_case_skip_comment == 0u;
           emel_case_skip_comment = 2u) {

      }
    }
    scan_more = static_cast<uint32_t>(advance_space | skip_comment);
  }
  return pos;
}

inline bool has_prefix(std::string_view input, uint32_t pos, std::string_view prefix) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const size_t in_bounds = static_cast<size_t>(pos + prefix.size() <= size);
  const uint32_t safe_pos = pos * static_cast<uint32_t>(in_bounds);
  const size_t safe_size = prefix.size() * in_bounds;
  return in_bounds != 0 && input.substr(safe_pos, safe_size) == prefix;
}

inline uint32_t scan_quoted(std::string_view input, uint32_t pos, const char terminator) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  ++pos;  // opening quote/bracket already consumed by caller.
  uint32_t matched = 0;
  while (pos < size && matched == 0u) {
    const char c = input[pos];
    const size_t escaped = static_cast<size_t>(c == '\\' && pos + 1u < size);
    pos += static_cast<uint32_t>(escaped + 1u);
    matched = static_cast<uint32_t>(static_cast<size_t>(c == terminator) & (1u - escaped));
  }
  return pos;
}

inline uint32_t scan_braced_quantifier(std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  ++pos;  // consume '{'
  while (pos < size && input[pos] != '}') {
    ++pos;
  }
  pos += static_cast<uint32_t>(pos < size && input[pos] == '}');
  return pos;
}

inline uint32_t scan_token_ref(std::string_view input, uint32_t pos) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  pos += static_cast<uint32_t>(input[pos] == '!');
  const size_t has_open = static_cast<size_t>(pos + 1u < size && input[pos] == '<' &&
                                              input[pos + 1u] == '[');
  {
    const size_t emel_branch_has_open = has_open;
    for (size_t emel_case_has_open = emel_branch_has_open; emel_case_has_open == 0u;
         emel_case_has_open = 2u) {
      return pos;
    }
    for (size_t emel_case_has_open = emel_branch_has_open; emel_case_has_open == 1u;
         emel_case_has_open = 2u) {
      pos += 2u;
      while (pos < size && input[pos] >= '0' && input[pos] <= '9') {
        ++pos;
      }
      const size_t is_closed = static_cast<size_t>(pos + 1u < size && input[pos] == ']' &&
                                                   input[pos + 1u] == '>');
      const uint32_t end_positions[2] = {pos, static_cast<uint32_t>(pos + 2u)};
      return end_positions[is_closed];
    }
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

  const size_t at_end = static_cast<size_t>(pos >= size);
  const uint32_t offset_candidates[2] = {next_offset, pos};
  next_offset = offset_candidates[at_end];
  {
    const size_t emel_branch_at_end = at_end;
    for (size_t emel_case_at_end = emel_branch_at_end; emel_case_at_end == 1u;
         emel_case_at_end = 2u) {
      return event::token{};
    }
    for (size_t emel_case_at_end = emel_branch_at_end; emel_case_at_end == 0u;
         emel_case_at_end = 2u) {

    }
  }

  const uint32_t start = pos;
  const char c = input[pos];

  const size_t newline = static_cast<size_t>(is_newline_char(c));
  {
    const size_t emel_branch_newline = newline;
    for (size_t emel_case_newline = emel_branch_newline; emel_case_newline == 1u;
         emel_case_newline = 2u) {
      const size_t crlf = static_cast<size_t>(c == '\r' && pos + 1u < size &&
                                              input[pos + 1u] == '\n');
      const uint32_t newline_steps[2] = {1u, 2u};
      pos += newline_steps[crlf];
      next_offset = pos;
      return make_token(input, start, pos, event::token_kind::newline);
    }
    for (size_t emel_case_newline = emel_branch_newline; emel_case_newline == 0u;
         emel_case_newline = 2u) {

    }
  }

  const size_t definition = static_cast<size_t>(has_prefix(input, pos, "::="));
  {
    const size_t emel_branch_definition = definition;
    for (size_t emel_case_definition = emel_branch_definition; emel_case_definition == 1u;
         emel_case_definition = 2u) {
      pos += 3u;
      next_offset = pos;
      return make_token(input, start, pos, event::token_kind::definition_operator);
    }
    for (size_t emel_case_definition = emel_branch_definition; emel_case_definition == 0u;
         emel_case_definition = 2u) {

    }
  }

  const size_t is_alternation = static_cast<size_t>(c == '|');
  const size_t is_dot = static_cast<size_t>(c == '.');
  const size_t is_open_group = static_cast<size_t>(c == '(');
  const size_t is_close_group = static_cast<size_t>(c == ')');
  const size_t is_simple_quantifier =
      static_cast<size_t>(c == '+' || c == '*' || static_cast<unsigned char>(c) == 63u);
  const size_t is_string_literal = static_cast<size_t>(c == '"');
  const size_t is_character_class = static_cast<size_t>(c == '[');
  const size_t is_braced_quantifier = static_cast<size_t>(c == '{');
  const size_t symbol_mode = is_alternation * 1u + is_dot * 2u + is_open_group * 3u +
                             is_close_group * 4u + is_simple_quantifier * 5u +
                             is_string_literal * 6u + is_character_class * 7u +
                             is_braced_quantifier * 8u;

  {
    const size_t emel_branch_symbol = static_cast<size_t>(symbol_mode != 0u);
    for (size_t emel_case_symbol = emel_branch_symbol; emel_case_symbol == 1u;
         emel_case_symbol = 2u) {
      const uint32_t one_char_end = static_cast<uint32_t>(pos + 1u);
      uint32_t token_end = one_char_end;
      {
        const size_t emel_branch_string = static_cast<size_t>(symbol_mode == 6u);
        for (size_t emel_case_string = emel_branch_string; emel_case_string == 1u;
             emel_case_string = 2u) {
          token_end = scan_quoted(input, pos, '"');
        }
        for (size_t emel_case_string = emel_branch_string; emel_case_string == 0u;
             emel_case_string = 2u) {

        }
      }
      {
        const size_t emel_branch_class = static_cast<size_t>(symbol_mode == 7u);
        for (size_t emel_case_class = emel_branch_class; emel_case_class == 1u;
             emel_case_class = 2u) {
          token_end = scan_quoted(input, pos, ']');
        }
        for (size_t emel_case_class = emel_branch_class; emel_case_class == 0u;
             emel_case_class = 2u) {

        }
      }
      {
        const size_t emel_branch_braced = static_cast<size_t>(symbol_mode == 8u);
        for (size_t emel_case_braced = emel_branch_braced; emel_case_braced == 1u;
             emel_case_braced = 2u) {
          token_end = scan_braced_quantifier(input, pos);
        }
        for (size_t emel_case_braced = emel_branch_braced; emel_case_braced == 0u;
             emel_case_braced = 2u) {

        }
      }

      constexpr event::token_kind kinds[9] = {
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

      pos = token_end;
      next_offset = pos;
      return make_token(input, start, pos, kinds[symbol_mode]);
    }
    for (size_t emel_case_symbol = emel_branch_symbol; emel_case_symbol == 0u;
         emel_case_symbol = 2u) {

    }
  }

  const size_t starts_rule_ref = static_cast<size_t>(
      c == '<' || (c == '!' && has_prefix(input, pos + 1u, "<[")));
  uint32_t rule_ref_end = pos;
  {
    const size_t emel_branch_rule_ref = starts_rule_ref;
    for (size_t emel_case_rule_ref = emel_branch_rule_ref; emel_case_rule_ref == 1u;
         emel_case_rule_ref = 2u) {
      rule_ref_end = scan_token_ref(input, pos);
    }
    for (size_t emel_case_rule_ref = emel_branch_rule_ref; emel_case_rule_ref == 0u;
         emel_case_rule_ref = 2u) {

    }
  }
  const size_t matched_rule_ref =
      starts_rule_ref & static_cast<size_t>(rule_ref_end > pos);
  {
    const size_t emel_branch_matched_rule_ref = matched_rule_ref;
    for (size_t emel_case_matched_rule_ref = emel_branch_matched_rule_ref;
         emel_case_matched_rule_ref == 1u;
         emel_case_matched_rule_ref = 2u) {
      next_offset = rule_ref_end;
      return make_token(input, start, rule_ref_end, event::token_kind::rule_reference);
    }
    for (size_t emel_case_matched_rule_ref = emel_branch_matched_rule_ref;
         emel_case_matched_rule_ref == 0u;
         emel_case_matched_rule_ref = 2u) {

    }
  }

  const size_t is_word = static_cast<size_t>(is_word_char(c));
  {
    const size_t emel_branch_word = is_word;
    for (size_t emel_case_word = emel_branch_word; emel_case_word == 1u;
         emel_case_word = 2u) {
      ++pos;
      while (pos < size && is_word_char(input[pos])) {
        ++pos;
      }
      next_offset = pos;
      return make_token(input, start, pos, event::token_kind::identifier);
    }
    for (size_t emel_case_word = emel_branch_word; emel_case_word == 0u;
         emel_case_word = 2u) {

    }
  }

  ++pos;
  next_offset = pos;
  return make_token(input, start, pos, event::token_kind::unknown);
}

inline bool noop_error_callback(const events::next_error &) noexcept {
  return true;
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
      const size_t has_callback = static_cast<size_t>(static_cast<bool>(ev.on_error));
      const callback<bool(const events::next_error &)> callbacks[2] = {
          callback<bool(const events::next_error &)>::from<detail::noop_error_callback>(),
          ev.on_error};
      (void)callbacks[has_callback](events::next_error{error_code(error::internal_error)});
    }
  }
};

inline constexpr emit_next_token emit_next_token{};
inline constexpr emit_eof emit_eof{};
inline constexpr reject_invalid_next reject_invalid_next{};
inline constexpr reject_invalid_cursor reject_invalid_cursor{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::lexer::action
