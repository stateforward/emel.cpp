#pragma once

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/lexer/actions.hpp"
#include "emel/text/jinja/parser/lexer/context.hpp"
#include "emel/text/jinja/parser/lexer/detail.hpp"
#include "emel/text/jinja/parser/lexer/guards.hpp"

namespace emel::text::jinja::parser::lexer {

struct initialized {};
struct scanning {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialized.
        sml::state<initialized> <= *sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::invalid_next{} ]
          / action::reject_invalid_next

      , sml::state<initialized> <= sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::invalid_cursor_position{} ]
          / action::reject_invalid_cursor

      , sml::state<initialized> <= sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::scan_failed{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::scan_has_token{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::scan_at_eof{} ]
          / action::emit_eof

      //------------------------------------------------------------------------------//
      // Scanning.
      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::invalid_next{} ]
          / action::reject_invalid_next

      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::invalid_cursor_position{} ]
          / action::reject_invalid_cursor

      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::scan_failed{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::scan_has_token{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::scan_at_eof{} ]
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

using sm = emel::sm<model, action::context>;
using Lexer = sm;

} // namespace emel::text::jinja::parser::lexer
