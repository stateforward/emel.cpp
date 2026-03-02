#pragma once

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
#include "emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"

namespace emel::text::jinja::parser::program_parser::expression_parser {

struct deciding {};
struct expression_first_decision {};
struct expression_scan {};
struct expression_emit_decision {};
struct expression_close {};
struct parsed {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Expression parser.
        sml::state<expression_first_decision> <= *sml::state<deciding>
          + sml::completion<event::parse_runtime>
          / action::begin_expression_parse

      , sml::state<parse_failed> <= sml::state<expression_first_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_scan_eof{} ]
          / action::fail_expression_start_token

      , sml::state<parse_failed> <= sml::state<expression_first_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_first_is_close{} ]
          / action::fail_expression_close_token

      , sml::state<expression_scan> <= sml::state<expression_first_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_first_is_identifier{} ]
          / action::consume_expression_identifier

      , sml::state<expression_scan> <= sml::state<expression_first_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_first_is_literal{} ]
          / action::consume_expression_literal

      , sml::state<expression_scan> <= sml::state<expression_first_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_first_is_unary{} ]
          / action::consume_expression_unary

      , sml::state<expression_scan> <= sml::state<expression_first_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_first_is_other_content{} ]
          / action::consume_expression_compound

      , sml::state<expression_emit_decision> <= sml::state<expression_scan>
          + sml::completion<event::parse_runtime>[ guard::expr_scan_at_close{} ]

      , sml::state<expression_scan> <= sml::state<expression_scan>
          + sml::completion<event::parse_runtime>[ guard::expr_scan_continue{} ]
          / action::consume_expression_token

      , sml::state<parse_failed> <= sml::state<expression_scan>
          + sml::completion<event::parse_runtime>[ guard::expr_scan_eof{} ]
          / action::fail_expression_start_token

      , sml::state<expression_close> <= sml::state<expression_emit_decision>
          + sml::completion<event::parse_runtime>[ guard::expression_identifier{} ]
          / action::emit_expression_identifier

      , sml::state<expression_close> <= sml::state<expression_emit_decision>
          + sml::completion<event::parse_runtime>[ guard::expression_non_identifier{} ]
          / action::emit_expression_generic

      , sml::state<parsed> <= sml::state<expression_close>
          + sml::completion<event::parse_runtime>[ guard::expr_scan_at_close{} ]
          / action::consume_expression_close

      , sml::state<parse_failed> <= sml::state<expression_close>
          + sml::completion<event::parse_runtime>[ guard::expr_scan_eof{} ]
          / action::fail_expression_start_token

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<expression_first_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<expression_scan> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<expression_emit_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<expression_close> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parsed> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parse_failed> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<unexpected_event> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : emel::sm<model> {
  using model_type = model;
};

} // namespace emel::text::jinja::parser::program_parser::expression_parser
