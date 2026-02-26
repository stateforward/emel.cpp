#pragma once

#include "emel/gbnf/definition_parser/context.hpp"
#include "emel/gbnf/definition_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::definition_parser::guard {

struct token_definition_operator {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(rule_parser::error::none) &&
           ev.flow.has_token &&
           ev.flow.token.kind == emel::gbnf::lexer::event::token_kind::definition_operator;
  }
};

struct parse_failed {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return ev.flow.err == emel::error::cast(rule_parser::error::none) &&
           !token_definition_operator{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::definition_parser::guard
