#pragma once

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
                 [ guard::has_remaining_input{} ]
                 / action::emit_next_token

      , sml::state<scanning> <= sml::state<initialized> + sml::event<event::next>
                 [ guard::at_eof{} ]
                 / action::emit_eof

      //------------------------------------------------------------------------------//
      // Scanning.
      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::invalid_next{} ]
                 / action::reject_invalid_next

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::invalid_cursor_position{} ]
                 / action::reject_invalid_cursor

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::has_remaining_input{} ]
                 / action::emit_next_token

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::next>
                 [ guard::at_eof{} ]
                 / action::emit_eof

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

using sm = emel::sm_with_context<model, action::context>;

}  // namespace emel::gbnf::rule_parser::lexer
