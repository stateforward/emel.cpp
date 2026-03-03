#pragma once

#include "emel/gbnf/rule_parser/lexer/context.hpp"
#include "emel/gbnf/rule_parser/lexer/detail.hpp"
#include "emel/gbnf/rule_parser/lexer/errors.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"

namespace emel::gbnf::rule_parser::lexer::guard {

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
    const bool in_range =
        valid_cursor_position{}(ev, ctx) && ev.cursor.offset < ev.cursor.input.size();
    const uint32_t start = detail::token_start(ev.cursor);
    return in_range && start >= ev.cursor.input.size();
  }
};

template <size_t mode_value>
struct symbol_mode_is {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = valid_cursor_position{}(ev, ctx) &&
                           detail::token_start(ev.cursor) < ev.cursor.input.size();
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::token_start(ev.cursor);
    const char c = ev.cursor.input[static_cast<size_t>(start)];
    return has_input && detail::symbol_mode(c) == mode_value;
  }
};

using starts_alternation = symbol_mode_is<1u>;
using starts_dot = symbol_mode_is<2u>;
using starts_open_group = symbol_mode_is<3u>;
using starts_close_group = symbol_mode_is<4u>;
using starts_quantifier = symbol_mode_is<5u>;
using starts_string_literal = symbol_mode_is<6u>;
using starts_character_class = symbol_mode_is<7u>;
using starts_braced_quantifier = symbol_mode_is<8u>;

struct starts_newline {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = valid_cursor_position{}(ev, ctx) &&
                           detail::token_start(ev.cursor) < ev.cursor.input.size();
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::token_start(ev.cursor);
    return has_input && detail::is_newline_char(ev.cursor.input[static_cast<size_t>(start)]);
  }
};

struct starts_definition_operator {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = valid_cursor_position{}(ev, ctx) &&
                           detail::token_start(ev.cursor) < ev.cursor.input.size();
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::token_start(ev.cursor);
    return has_input && detail::has_prefix(ev.cursor.input, start, "::=");
  }
};

struct starts_rule_reference {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = valid_cursor_position{}(ev, ctx) &&
                           detail::token_start(ev.cursor) < ev.cursor.input.size();
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::token_start(ev.cursor);
    const bool starts_ref = has_input && detail::starts_rule_reference(ev.cursor.input, start);
    const uint32_t end = detail::scan_token_ref(ev.cursor.input, start);
    return starts_ref && end > start;
  }
};

struct starts_identifier {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = valid_cursor_position{}(ev, ctx) &&
                           detail::token_start(ev.cursor) < ev.cursor.input.size();
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::token_start(ev.cursor);
    return has_input && detail::is_word_char(ev.cursor.input[static_cast<size_t>(start)]);
  }
};

struct starts_unknown {
  bool operator()(const event::next & ev, const action::context & ctx) const noexcept {
    const bool has_input = valid_cursor_position{}(ev, ctx) &&
                           detail::token_start(ev.cursor) < ev.cursor.input.size();
    if (!has_input) {
      return false;
    }
    const uint32_t start = detail::token_start(ev.cursor);
    const char c = ev.cursor.input[static_cast<size_t>(start)];
    const size_t known_mode = static_cast<size_t>(detail::symbol_mode(c) != 0u);
    const size_t known_newline = static_cast<size_t>(detail::is_newline_char(c));
    const size_t known_definition = static_cast<size_t>(detail::has_prefix(ev.cursor.input, start, "::="));
    const size_t starts_ref = static_cast<size_t>(detail::starts_rule_reference(ev.cursor.input, start));
    const size_t matched_ref =
        starts_ref & static_cast<size_t>(detail::scan_token_ref(ev.cursor.input, start) > start);
    const size_t known_word = static_cast<size_t>(detail::is_word_char(c));
    const size_t known = known_mode | known_newline | known_definition | matched_ref | known_word;
    return has_input && known == 0u;
  }
};

}  // namespace emel::gbnf::rule_parser::lexer::guard
