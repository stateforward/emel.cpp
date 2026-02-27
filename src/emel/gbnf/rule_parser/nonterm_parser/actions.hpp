#pragma once

#include <cstdint>

#include "emel/gbnf/rule_parser/nonterm_parser/context.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::rule_parser::nonterm_parser::action {

struct consume_definition_existing {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.ctx.token.text);
    uint32_t rule_id = 0;
    (void)ctx.symbols.find(ev.ctx.token.text, hash, rule_id);
    ctx.rule_defined[rule_id] = true;
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = rule_id;
  }
};

struct consume_definition_new {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.ctx.token.text);
    const uint32_t rule_id = ctx.next_symbol_id++;
    (void)ctx.symbols.insert(ev.ctx.token.text, hash, rule_id);
    ctx.rule_defined[rule_id] = true;
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = rule_id;
  }
};

struct consume_reference_existing {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.ctx.token.text);
    uint32_t rule_id = 0;
    (void)ctx.symbols.find(ev.ctx.token.text, hash, rule_id);
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = rule_id;
  }
};

struct consume_reference_new {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.ctx.token.text);
    const uint32_t rule_id = ctx.next_symbol_id++;
    (void)ctx.symbols.insert(ev.ctx.token.text, hash, rule_id);
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = rule_id;
  }
};

struct dispatch_parse_failed {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(rule_parser::error::parse_failed);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(rule_parser::error::internal_error);
    }
  }
};

inline constexpr consume_definition_existing consume_definition_existing{};
inline constexpr consume_definition_new consume_definition_new{};
inline constexpr consume_reference_existing consume_reference_existing{};
inline constexpr consume_reference_new consume_reference_new{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::nonterm_parser::action
