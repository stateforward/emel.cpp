#pragma once

#include "emel/gbnf/rule_parser/expression_parser/context.hpp"
#include "emel/gbnf/rule_parser/expression_parser/events.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::rule_parser::expression_parser::action {

struct consume_identifier {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.expression_kind = events::parse_kind::identifier;
  }
};

struct consume_non_identifier {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(rule_parser::error::none);
    ev.ctx.expression_kind = events::parse_kind::non_identifier;
  }
};

struct dispatch_parse_failed {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.ctx.err = emel::error::cast(rule_parser::error::parse_failed);
    ev.ctx.expression_kind = events::parse_kind::unknown;
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

inline constexpr consume_identifier consume_identifier{};
inline constexpr consume_non_identifier consume_non_identifier{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::rule_parser::expression_parser::action
