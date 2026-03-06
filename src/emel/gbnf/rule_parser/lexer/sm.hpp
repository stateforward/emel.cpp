#pragma once
// benchmark: designed

#include "emel/gbnf/rule_parser/lexer/actions.hpp"
#include "emel/gbnf/rule_parser/lexer/events.hpp"
#include "emel/gbnf/rule_parser/lexer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::rule_parser::lexer {

struct initialized {};
struct scanning {};
struct scan_ready {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<initialized> <= *sml::state<initialized> + sml::event<event::scan_next>
                 [ guard::invalid_next{} ]
                 / action::reject_invalid_next

      , sml::state<initialized> <= sml::state<initialized> + sml::event<event::scan_next>
                 [ guard::invalid_cursor_position{} ]
                 / action::reject_invalid_cursor

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::scan_next>
                 [ guard::invalid_next{} ]
                 / action::reject_invalid_next

      , sml::state<scanning> <= sml::state<scanning> + sml::event<event::scan_next>
                 [ guard::invalid_cursor_position{} ]
                 / action::reject_invalid_cursor

      , sml::state<scan_ready> <= sml::state<initialized> + sml::event<event::scan_next>
                 [ guard::valid_cursor_position{} ]
                 / action::prepare_scan

      , sml::state<scan_ready> <= sml::state<scanning> + sml::event<event::scan_next>
                 [ guard::valid_cursor_position{} ]
                 / action::prepare_scan

      //------------------------------------------------------------------------------//
      // Token dispatch.
      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::at_eof{} ]
                 / action::emit_eof

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::layout_exhausted{} ]
                 / action::emit_layout_exhausted_unknown

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_newline_crlf{} ]
                 / action::emit_newline_crlf_token

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_newline_single{} ]
                 / action::emit_newline_single_token

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_definition_operator{} ]
                 / action::emit_definition_operator

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_alternation{} ]
                 / action::emit_alternation

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_dot{} ]
                 / action::emit_dot

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_open_group{} ]
                 / action::emit_open_group

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_close_group{} ]
                 / action::emit_close_group

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_quantifier{} ]
                 / action::emit_quantifier

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_string_literal{} ]
                 / action::emit_string_literal

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_character_class{} ]
                 / action::emit_character_class

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_braced_quantifier{} ]
                 / action::emit_braced_quantifier

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::parsed_rule_reference_negated_valid{} ]
                 / action::emit_rule_reference_negated

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::parsed_rule_reference_negated_invalid{} ]
                 / action::emit_unknown

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::parsed_rule_reference_plain_valid{} ]
                 / action::emit_rule_reference_plain

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::parsed_rule_reference_plain_invalid{} ]
                 / action::emit_unknown

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 [ guard::starts_identifier{} ]
                 / action::emit_identifier

      , sml::state<scanning> <= sml::state<scan_ready> + sml::completion<event::scan_next>
                 / action::emit_unknown

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<initialized> <= sml::state<initialized> + sml::unexpected_event<sml::_>
                 [ guard::unexpected_has_error_callback{} ]
                 / action::dispatch_unexpected_error

      , sml::state<scanning> <= sml::state<scanning> + sml::unexpected_event<sml::_>
                 [ guard::unexpected_has_error_callback{} ]
                 / action::dispatch_unexpected_error

      , sml::state<scan_ready> <= sml::state<scan_ready> + sml::unexpected_event<sml::_>
                 [ guard::unexpected_has_error_callback{} ]
                 / action::dispatch_unexpected_error

      , sml::state<initialized> <= sml::state<initialized> + sml::unexpected_event<sml::_>
                 / action::ignore_unexpected

      , sml::state<scanning> <= sml::state<scanning> + sml::unexpected_event<sml::_>
                 / action::ignore_unexpected

      , sml::state<scan_ready> <= sml::state<scan_ready> + sml::unexpected_event<sml::_>
                 / action::ignore_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;

  bool process_event(const event::next & ev) {
    event::scan_ctx ctx{};
    event::scan_next internal{ev, ctx};
    return base_type::process_event(internal);
  }
};

}  // namespace emel::gbnf::rule_parser::lexer
