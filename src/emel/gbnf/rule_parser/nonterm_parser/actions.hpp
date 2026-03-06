#pragma once

#include <cstdint>
#include <string_view>

#include "emel/gbnf/rule_parser/nonterm_parser/context.hpp"
#include "emel/gbnf/rule_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::rule_parser::nonterm_parser::action {

inline bool has_insert_slot(const context & ctx,
                            const std::string_view name,
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
    if (entry.hash == hash && entry.name == name) {
      return true;
    }
    slot = (slot + 1u) & mask;
  }
  return false;
}

struct lookup_definition_candidate {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context & ctx) const noexcept {
    const std::string_view text = ev.ctx.token.text;
    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(text);
    uint32_t rule_id = 0;
    const bool found = ctx.symbols.find(text, hash, rule_id);
    const bool can_insert =
        !found &&
        ctx.next_symbol_id < emel::gbnf::k_max_gbnf_rules &&
        ctx.symbols.count < emel::gbnf::k_max_gbnf_symbols &&
        has_insert_slot(ctx, text, hash);

    ev.ctx.nonterm_lookup_hash = hash;
    ev.ctx.nonterm_lookup_rule_id = rule_id;
    ev.ctx.nonterm_lookup_found = found;
    ev.ctx.nonterm_lookup_can_insert = can_insert;
  }
};

struct lookup_reference_candidate {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context & ctx) const noexcept {
    const std::string_view text = ev.ctx.token.text;
    const uint32_t hash = rule_parser::detail::symbol_table::hash_name(text);
    uint32_t rule_id = 0;
    const bool found = ctx.symbols.find(text, hash, rule_id);
    const bool can_insert =
        !found &&
        ctx.next_symbol_id < emel::gbnf::k_max_gbnf_rules &&
        ctx.symbols.count < emel::gbnf::k_max_gbnf_symbols &&
        has_insert_slot(ctx, text, hash);

    ev.ctx.nonterm_lookup_hash = hash;
    ev.ctx.nonterm_lookup_rule_id = rule_id;
    ev.ctx.nonterm_lookup_found = found;
    ev.ctx.nonterm_lookup_can_insert = can_insert;
  }
};

struct consume_definition_existing {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t rule_id = ev.ctx.nonterm_lookup_rule_id;
    ctx.rule_defined[rule_id] = true;
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = rule_id;
  }
};

struct consume_definition_new {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t rule_id = ctx.next_symbol_id++;
    (void)ctx.symbols.insert(ev.ctx.token.text, ev.ctx.nonterm_lookup_hash, rule_id);
    ctx.rule_defined[rule_id] = true;
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = rule_id;
  }
};

struct consume_reference_existing {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.nonterm_rule_id = ev.ctx.nonterm_lookup_rule_id;
  }
};

struct consume_reference_new {
  void operator()(const rule_parser::event::parse_rules & ev,
                  context & ctx) const noexcept {
    const uint32_t rule_id = ctx.next_symbol_id++;
    (void)ctx.symbols.insert(ev.ctx.token.text, ev.ctx.nonterm_lookup_hash, rule_id);
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

inline constexpr lookup_definition_candidate lookup_definition_candidate{};
inline constexpr lookup_reference_candidate lookup_reference_candidate{};
inline constexpr consume_definition_existing consume_definition_existing{};
inline constexpr consume_definition_new consume_definition_new{};
inline constexpr consume_reference_existing consume_reference_existing{};
inline constexpr consume_reference_new consume_reference_new{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::nonterm_parser::action
