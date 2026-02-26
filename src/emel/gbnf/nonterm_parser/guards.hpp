#pragma once

#include <cstdint>

#include "emel/gbnf/nonterm_parser/context.hpp"
#include "emel/gbnf/nonterm_parser/events.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::nonterm_parser::guard {

inline bool has_insert_slot(const rule_parser::event::parse_rules & ev,
                            const action::context & ctx,
                            const uint32_t hash) noexcept {
  const auto & entries = ctx.symbols.entries;
  const uint32_t slot_count = static_cast<uint32_t>(entries.size());
  const uint32_t mask = slot_count - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0; probes < slot_count; ++probes) {
    const auto & entry = entries[slot];
    if (!entry.occupied) {
      return true;
    }
    if (entry.hash == hash && entry.name == ev.flow.token.text) {
      return true;
    }
    slot = (slot + 1u) & mask;
  }
  return false;
}

struct token_identifier {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.flow.err == emel::error::cast(rule_parser::error::none) &&
           ev.flow.has_token &&
           ev.flow.token.kind == emel::gbnf::lexer::event::token_kind::identifier &&
           !ev.flow.token.text.empty();
  }
};

struct mode_definition {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.flow.nonterm_mode == events::parse_mode::definition;
  }
};

struct mode_reference {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context &) const noexcept {
    return ev.flow.nonterm_mode == events::parse_mode::reference;
  }
};

struct definition_existing_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    if (!token_identifier{}(ev, ctx) || !mode_definition{}(ev, ctx)) {
      return false;
    }

    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.flow.token.text);
    uint32_t id = 0;
    if (!ctx.symbols.find(ev.flow.token.text, hash, id)) {
      return false;
    }
    return id < ctx.rule_defined.size() && !ctx.rule_defined[id];
  }
};

struct definition_new_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    if (!token_identifier{}(ev, ctx) || !mode_definition{}(ev, ctx)) {
      return false;
    }

    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.flow.token.text);
    uint32_t id = 0;
    if (ctx.symbols.find(ev.flow.token.text, hash, id)) {
      return false;
    }
    if (ctx.next_symbol_id >= emel::gbnf::k_max_gbnf_rules ||
        ctx.symbols.count >= emel::gbnf::k_max_gbnf_symbols) {
      return false;
    }

    return has_insert_slot(ev, ctx, hash);
  }
};

struct reference_existing_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    if (!token_identifier{}(ev, ctx) || !mode_reference{}(ev, ctx)) {
      return false;
    }

    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.flow.token.text);
    uint32_t id = 0;
    return ctx.symbols.find(ev.flow.token.text, hash, id);
  }
};

struct reference_new_valid {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    if (!token_identifier{}(ev, ctx) || !mode_reference{}(ev, ctx)) {
      return false;
    }

    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(ev.flow.token.text);
    uint32_t id = 0;
    if (ctx.symbols.find(ev.flow.token.text, hash, id)) {
      return false;
    }
    if (ctx.next_symbol_id >= emel::gbnf::k_max_gbnf_rules ||
        ctx.symbols.count >= emel::gbnf::k_max_gbnf_symbols) {
      return false;
    }

    return has_insert_slot(ev, ctx, hash);
  }
};

struct parse_failed {
  bool operator()(const rule_parser::event::parse_rules & ev,
                  const action::context & ctx) const noexcept {
    return ev.flow.err == emel::error::cast(rule_parser::error::none) &&
           !definition_existing_valid{}(ev, ctx) &&
           !definition_new_valid{}(ev, ctx) &&
           !reference_existing_valid{}(ev, ctx) &&
           !reference_new_valid{}(ev, ctx);
  }
};

}  // namespace emel::gbnf::nonterm_parser::guard
