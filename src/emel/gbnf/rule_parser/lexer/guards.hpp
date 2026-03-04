#pragma once

#include "emel/gbnf/rule_parser/lexer/context.hpp"
#include "emel/gbnf/rule_parser/lexer/detail.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"

namespace emel::gbnf::rule_parser::lexer::guard {

namespace detail {

inline bool has_prefix(const std::string_view input,
                       const uint32_t pos,
                       const std::string_view prefix) noexcept {
  const uint32_t size = static_cast<uint32_t>(input.size());
  const size_t in_bounds = static_cast<size_t>(pos + prefix.size() <= size);
  const uint32_t safe_pos = pos * static_cast<uint32_t>(in_bounds);
  const size_t safe_size = prefix.size() * in_bounds;
  return in_bounds != 0u && input.substr(safe_pos, safe_size) == prefix;
}

inline bool has_scan_char(const event::next & ev, const action::context & ctx) noexcept;
inline uint32_t scan_start(const event::next & ev) noexcept;
inline char scan_char(const event::next & ev) noexcept;

}  // namespace detail

struct valid_next {
  bool operator()(const event::next & ev, const action::context &) const noexcept {
    return ev.on_done && ev.on_error;
  }
};

struct invalid_next {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return !valid_next{}(ev, ctx);
  }
};

struct valid_cursor_position {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return valid_next{}(ev, ctx) && ev.cursor.offset <= ev.cursor.input.size();
  }
};

struct invalid_cursor_position {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return valid_next{}(ev, ctx) && !valid_cursor_position{}(ev, ctx);
  }
};

struct has_remaining_input {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return valid_cursor_position{}(ev, ctx) && ev.cursor.offset < ev.cursor.input.size();
  }
};

struct at_eof {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return valid_cursor_position{}(ev, ctx) && ev.cursor.offset >= ev.cursor.input.size();
  }
};

struct layout_exhausted {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool in_range = valid_cursor_position{}(ev, ctx) && ev.cursor.offset < ev.cursor.input.size();
    const uint32_t start = lexer::detail::token_start(ev.cursor);
    return in_range && start >= ev.cursor.input.size();
  }
};

namespace detail {

inline bool has_scan_char(const event::next & ev, const action::context & ctx) noexcept {
  return valid_cursor_position{}(ev, ctx) && lexer::detail::token_start(ev.cursor) < ev.cursor.input.size();
}

inline uint32_t scan_start(const event::next & ev) noexcept {
  return lexer::detail::token_start(ev.cursor);
}

inline char scan_char(const event::next & ev) noexcept {
  return ev.cursor.input[static_cast<size_t>(scan_start(ev))];
}

}  // namespace detail

template <char symbol>
struct starts_symbol {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    return detail::scan_char(ev) == symbol;
  }
};

using starts_alternation = starts_symbol<'|'>;
using starts_dot = starts_symbol<'.'>;
using starts_open_group = starts_symbol<'('>;
using starts_close_group = starts_symbol<')'>;
using starts_string_literal = starts_symbol<'"'>;
using starts_character_class = starts_symbol<'['>;
using starts_braced_quantifier = starts_symbol<'{'>;

struct starts_quantifier {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    const char c = detail::scan_char(ev);
    const size_t plus = static_cast<size_t>(c == '+');
    const size_t star = static_cast<size_t>(c == '*');
    const size_t question = static_cast<size_t>(static_cast<unsigned char>(c) == 63u);
    return (plus | star | question) != 0u;
  }
};

struct starts_newline {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    const char c = detail::scan_char(ev);
    return c == '\n' || c == '\r';
  }
};

struct starts_newline_crlf {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::scan_start(ev);
    return detail::has_prefix(ev.cursor.input, start, "\r\n");
  }
};

struct starts_newline_single {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return starts_newline{}(ev, ctx) && !starts_newline_crlf{}(ev, ctx);
  }
};

struct starts_definition_operator {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::scan_start(ev);
    return detail::has_prefix(ev.cursor.input, start, "::=");
  }
};

struct starts_rule_reference_plain_candidate {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::scan_start(ev);
    return detail::has_prefix(ev.cursor.input, start, "<[");
  }
};

struct starts_rule_reference_negated_candidate {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::scan_start(ev);
    const bool has_bang = detail::scan_char(ev) == '!';
    return has_bang && detail::has_prefix(ev.cursor.input, start + 1u, "<[");
  }
};

struct parsed_rule_reference_plain_valid {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const uint32_t start = detail::scan_start(ev);
    const uint32_t end = lexer::detail::scan_token_ref_plain(ev.cursor.input, start);
    return starts_rule_reference_plain_candidate{}(ev, ctx) && end > start;
  }
};

struct parsed_rule_reference_plain_invalid {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return starts_rule_reference_plain_candidate{}(ev, ctx) &&
           !parsed_rule_reference_plain_valid{}(ev, ctx);
  }
};

struct parsed_rule_reference_negated_valid {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const uint32_t start = detail::scan_start(ev);
    const uint32_t end = lexer::detail::scan_token_ref_plain(ev.cursor.input, start + 1u);
    return starts_rule_reference_negated_candidate{}(ev, ctx) && end > (start + 1u);
  }
};

struct parsed_rule_reference_negated_invalid {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    return starts_rule_reference_negated_candidate{}(ev, ctx) &&
           !parsed_rule_reference_negated_valid{}(ev, ctx);
  }
};

struct starts_identifier {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = detail::has_scan_char(ev, ctx);
    if (!has_input) {
      return false;
    }
    return lexer::detail::is_word_char(detail::scan_char(ev));
  }
};

struct unexpected_has_error_callback {
  template <class event_type>
  bool operator()(const event_type & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.on_error; }) {
      return static_cast<bool>(ev.on_error);
    }
    return false;
  }
};

}  // namespace emel::gbnf::rule_parser::lexer::guard
