#pragma once

#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/term_parser/context.hpp"

namespace emel::gbnf::rule_parser::term_parser::guard {

inline bool token_is(const rule_parser::event::parse_rules & ev,
                     const emel::gbnf::rule_parser::lexer::event::token_kind kind) noexcept {
  return ev.ctx.err == emel::error::cast(rule_parser::error::none) &&
         ev.ctx.has_token &&
         ev.ctx.token.kind == kind;
}

struct token_string_literal {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::string_literal);
  }
};

struct token_character_class {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::character_class);
  }
};

struct token_rule_reference {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::rule_reference);
  }
};

struct token_dot {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::dot);
  }
};

struct token_open_group {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::open_group);
  }
};

struct token_close_group {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::close_group);
  }
};

struct token_quantifier {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::quantifier);
  }
};

struct token_alternation {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::alternation);
  }
};

struct token_newline {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return token_is(ev, emel::gbnf::rule_parser::lexer::event::token_kind::newline);
  }
};

struct parse_failed {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(rule_parser::error::none) &&
           !token_string_literal{}(ev, ctx) &&
           !token_character_class{}(ev, ctx) &&
           !token_rule_reference{}(ev, ctx) &&
           !token_dot{}(ev, ctx) &&
           !token_open_group{}(ev, ctx) &&
           !token_close_group{}(ev, ctx) &&
           !token_quantifier{}(ev, ctx) &&
           !token_alternation{}(ev, ctx) &&
           !token_newline{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::rule_parser::term_parser::guard
