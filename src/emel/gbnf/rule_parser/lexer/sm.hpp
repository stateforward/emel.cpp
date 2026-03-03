#pragma once
// benchmark: designed

#include "emel/gbnf/rule_parser/lexer/actions.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"
#include "emel/gbnf/rule_parser/lexer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::rule_parser::lexer {

struct initialized {};
struct scanning {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialized.
        sml::state<initialized> <= *sml::state<initialized> + sml::event<event::next>
                 [ guard::invalid_next{} ]
                 / action::reject_invalid_next

      , sml::state<initialized> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::invalid_cursor_position{} ]
                 / action::reject_invalid_cursor

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::at_eof{} ]
                 / action::emit_eof

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::layout_exhausted{} ]
                 / action::emit_layout_exhausted_unknown

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_newline{} ]
                 / action::emit_newline_token

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_definition_operator{} ]
                 / action::emit_definition_operator

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_alternation{} ]
                 / action::emit_alternation

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_dot{} ]
                 / action::emit_dot

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_open_group{} ]
                 / action::emit_open_group

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_close_group{} ]
                 / action::emit_close_group

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_quantifier{} ]
                 / action::emit_quantifier

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_string_literal{} ]
                 / action::emit_string_literal

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_character_class{} ]
                 / action::emit_character_class

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_braced_quantifier{} ]
                 / action::emit_braced_quantifier

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_rule_reference{} ]
                 / action::emit_rule_reference

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_identifier{} ]
                 / action::emit_identifier

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::starts_unknown{} ]
                 / action::emit_unknown

      //------------------------------------------------------------------------------//
      // Scanning.
      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::invalid_next{} ]
                 / action::reject_invalid_next

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::invalid_cursor_position{} ]
                 / action::reject_invalid_cursor

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::at_eof{} ]
                 / action::emit_eof

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::layout_exhausted{} ]
                 / action::emit_layout_exhausted_unknown

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_newline{} ]
                 / action::emit_newline_token

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_definition_operator{} ]
                 / action::emit_definition_operator

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_alternation{} ]
                 / action::emit_alternation

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_dot{} ]
                 / action::emit_dot

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_open_group{} ]
                 / action::emit_open_group

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_close_group{} ]
                 / action::emit_close_group

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_quantifier{} ]
                 / action::emit_quantifier

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_string_literal{} ]
                 / action::emit_string_literal

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_character_class{} ]
                 / action::emit_character_class

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_braced_quantifier{} ]
                 / action::emit_braced_quantifier

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_rule_reference{} ]
                 / action::emit_rule_reference

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_identifier{} ]
                 / action::emit_identifier

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::starts_unknown{} ]
                 / action::emit_unknown

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<initialized> <= sml::state<initialized> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<scanning> <= sml::state<scanning> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

using sm = emel::sm<model, action::context>;

}  // namespace emel::gbnf::rule_parser::lexer
