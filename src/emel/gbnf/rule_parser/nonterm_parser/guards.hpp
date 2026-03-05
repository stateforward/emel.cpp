#pragma once

#include <cstdint>

#include "emel/gbnf/rule_parser/nonterm_parser/context.hpp"
#include "emel/gbnf/rule_parser/nonterm_parser/events.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::rule_parser::nonterm_parser::guard {

struct token_identifier_definition {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(rule_parser::error::none) &&
           ev.ctx.has_token &&
           ev.ctx.token.kind == emel::gbnf::rule_parser::lexer::event::token_kind::identifier &&
           !ev.ctx.token.text.empty() &&
           ev.ctx.nonterm_mode == events::parse_mode::definition;
  }
};

struct token_identifier_reference {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(rule_parser::error::none) &&
           ev.ctx.has_token &&
           ev.ctx.token.kind == emel::gbnf::rule_parser::lexer::event::token_kind::identifier &&
           !ev.ctx.token.text.empty() &&
           ev.ctx.nonterm_mode == events::parse_mode::reference;
  }
};

struct definition_existing_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.nonterm_lookup_found &&
           ev.ctx.nonterm_lookup_rule_id < ctx.rule_defined.size() &&
           !ctx.rule_defined[ev.ctx.nonterm_lookup_rule_id];
  }
};

struct definition_new_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return !ev.ctx.nonterm_lookup_found && ev.ctx.nonterm_lookup_can_insert;
  }
};

struct definition_failed {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return !definition_existing_valid{}(ev, ctx) &&
           !definition_new_valid{}(ev, ctx);
  }
};

struct reference_existing_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.ctx.nonterm_lookup_found;
  }
};

struct reference_new_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return !ev.ctx.nonterm_lookup_found && ev.ctx.nonterm_lookup_can_insert;
  }
};

struct reference_failed {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return !reference_existing_valid{}(ev, ctx) &&
           !reference_new_valid{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::rule_parser::nonterm_parser::guard
