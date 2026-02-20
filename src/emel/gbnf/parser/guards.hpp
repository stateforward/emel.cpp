#pragma once

#include "emel/gbnf/parser/context.hpp"
#include "emel/gbnf/parser/events.hpp"

namespace emel::gbnf::parser::guard {

struct valid_parse {
  bool operator()(const event::parse &ev,
                  const action::context &) const noexcept {
    if (ev.grammar_text.data() == nullptr || ev.grammar_text.empty()) {
      return false;
    }
    if (ev.grammar_out == nullptr || ev.error_out == nullptr) {
      return false;
    }
    return true;
  }
};

struct invalid_parse {
  bool operator()(const event::parse &ev,
                  const action::context &ctx) const noexcept {
    return !valid_parse{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

} // namespace emel::gbnf::parser::guard
