#pragma once

#include "emel/gbnf/definition_parser/context.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"

namespace emel::gbnf::definition_parser::action {

struct consume_definition_operator {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.flow.err = emel::error::cast(rule_parser::error::none);
  }
};

struct dispatch_parse_failed {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.flow.err = emel::error::cast(rule_parser::error::parse_failed);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.flow.err; }) {
      ev.flow.err = emel::error::cast(rule_parser::error::internal_error);
    }
  }
};

inline constexpr consume_definition_operator consume_definition_operator{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::definition_parser::action
