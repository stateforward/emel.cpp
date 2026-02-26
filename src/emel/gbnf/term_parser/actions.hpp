#pragma once

#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/term_parser/context.hpp"
#include "emel/gbnf/term_parser/events.hpp"

namespace emel::gbnf::term_parser::action {

template <events::term_kind kind>
struct consume_kind {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.flow.err = emel::error::cast(rule_parser::error::none);
    ev.flow.term_kind = kind;
  }
};

struct dispatch_parse_failed {
  void operator()(const rule_parser::event::parse_rules & ev,
                  const context &) const noexcept {
    ev.flow.err = emel::error::cast(rule_parser::error::parse_failed);
    ev.flow.term_kind = events::term_kind::unknown;
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

inline constexpr consume_kind<events::term_kind::string_literal> consume_string_literal{};
inline constexpr consume_kind<events::term_kind::character_class> consume_character_class{};
inline constexpr consume_kind<events::term_kind::rule_reference> consume_rule_reference{};
inline constexpr consume_kind<events::term_kind::dot> consume_dot{};
inline constexpr consume_kind<events::term_kind::open_group> consume_open_group{};
inline constexpr consume_kind<events::term_kind::close_group> consume_close_group{};
inline constexpr consume_kind<events::term_kind::quantifier> consume_quantifier{};
inline constexpr consume_kind<events::term_kind::alternation> consume_alternation{};
inline constexpr consume_kind<events::term_kind::newline> consume_newline{};
inline constexpr dispatch_parse_failed dispatch_parse_failed{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::gbnf::term_parser::action
