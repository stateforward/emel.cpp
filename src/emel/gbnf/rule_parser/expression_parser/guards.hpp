#pragma once

#include "emel/gbnf/rule_parser/expression_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::rule_parser::expression_parser::guard {

struct token_identifier {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(rule_parser::error::none) &&
           ev.ctx.has_token &&
           ev.ctx.token.kind == emel::gbnf::rule_parser::lexer::event::token_kind::identifier;
  }
};

struct token_non_identifier {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    if (ev.ctx.err != emel::error::cast(rule_parser::error::none) || !ev.ctx.has_token) {
      return false;
    }

    switch (ev.ctx.token.kind) {
      case emel::gbnf::rule_parser::lexer::event::token_kind::string_literal:
      case emel::gbnf::rule_parser::lexer::event::token_kind::character_class:
      case emel::gbnf::rule_parser::lexer::event::token_kind::rule_reference:
      case emel::gbnf::rule_parser::lexer::event::token_kind::dot:
      case emel::gbnf::rule_parser::lexer::event::token_kind::open_group:
      case emel::gbnf::rule_parser::lexer::event::token_kind::close_group:
      case emel::gbnf::rule_parser::lexer::event::token_kind::quantifier:
      case emel::gbnf::rule_parser::lexer::event::token_kind::alternation:
      case emel::gbnf::rule_parser::lexer::event::token_kind::newline:
        return true;
      default:
        return false;
    }
  }
};

struct parse_failed {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(rule_parser::error::none) &&
           !token_identifier{}(ev, ctx) &&
           !token_non_identifier{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::rule_parser::expression_parser::guard
