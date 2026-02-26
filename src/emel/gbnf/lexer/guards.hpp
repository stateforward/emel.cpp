#pragma once

#include "emel/gbnf/lexer/context.hpp"
#include "emel/gbnf/lexer/errors.hpp"
#include "emel/gbnf/lexer/events.hpp"

namespace emel::gbnf::lexer::guard {

struct valid_next {
  bool operator()(const event::next &ev, const action::context &) const noexcept {
    return ev.on_done && ev.on_error;
  }
};

struct invalid_next {
  bool operator()(const event::next &ev, const action::context &ctx) const noexcept {
    return !valid_next{}(ev, ctx);
  }
};

struct valid_cursor_position {
  bool operator()(const event::next &ev, const action::context &ctx) const noexcept {
    return valid_next{}(ev, ctx) && ev.cursor.offset <= ev.cursor.input.size();
  }
};

struct invalid_cursor_position {
  bool operator()(const event::next &ev, const action::context &ctx) const noexcept {
    return valid_next{}(ev, ctx) && !valid_cursor_position{}(ev, ctx);
  }
};

struct has_remaining_input {
  bool operator()(const event::next &ev, const action::context &ctx) const noexcept {
    return valid_cursor_position{}(ev, ctx) && ev.cursor.offset < ev.cursor.input.size();
  }
};

struct at_eof {
  bool operator()(const event::next &ev, const action::context &ctx) const noexcept {
    return valid_cursor_position{}(ev, ctx) && ev.cursor.offset >= ev.cursor.input.size();
  }
};

}  // namespace emel::gbnf::lexer::guard
