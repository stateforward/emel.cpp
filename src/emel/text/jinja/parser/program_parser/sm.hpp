#pragma once

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/actions.hpp"
#include "emel/text/jinja/parser/program_parser/expression_parser/sm.hpp"
#include "emel/text/jinja/parser/program_parser/guards.hpp"
#include "emel/text/jinja/parser/program_parser/statement_parser/sm.hpp"

namespace emel::text::jinja::parser::program_parser {

struct deciding {};
struct parse_begin {};
struct dispatch_decision {};
struct text_emit {};
struct comment_emit {};
struct statement_parse_result_decision {};
struct expression_parse_result_decision {};

struct parsed {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Program dispatch setup.
        sml::state<parse_begin> <= *sml::state<deciding>
          + sml::completion<event::parse_runtime>
          / action::start_program_parse

      , sml::state<dispatch_decision> <= sml::state<parse_begin>
          + sml::completion<event::parse_runtime>

      //------------------------------------------------------------------------------//
      // Node dispatch.
      , sml::state<parsed> <= sml::state<dispatch_decision>
          + sml::completion<event::parse_runtime>[ guard::at_eof{} ]
          / action::finish_parsed

      , sml::state<text_emit> <= sml::state<dispatch_decision>
          + sml::completion<event::parse_runtime>[ guard::token_text{} ]

      , sml::state<comment_emit> <= sml::state<dispatch_decision>
          + sml::completion<event::parse_runtime>[ guard::token_comment{} ]

      , sml::state<statement_parser::model> <= sml::state<dispatch_decision>
          + sml::completion<event::parse_runtime>[ guard::token_open_statement{} ]

      , sml::state<expression_parser::model> <= sml::state<dispatch_decision>
          + sml::completion<event::parse_runtime>[ guard::token_open_expression{} ]

      , sml::state<parse_failed> <= sml::state<dispatch_decision>
          + sml::completion<event::parse_runtime>[ guard::token_unexpected{} ]
          / action::fail_current_token

      //------------------------------------------------------------------------------//
      // Text/comment terminals.
      , sml::state<dispatch_decision> <= sml::state<text_emit>
          + sml::completion<event::parse_runtime>
          / action::consume_text

      , sml::state<dispatch_decision> <= sml::state<comment_emit>
          + sml::completion<event::parse_runtime>
          / action::consume_comment

      //------------------------------------------------------------------------------//
      // Statement and expression submachines.
      , sml::state<statement_parse_result_decision> <= sml::state<statement_parser::model>
          + sml::completion<event::parse_runtime>

      , sml::state<dispatch_decision> <= sml::state<statement_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_none{} ]

      , sml::state<parse_failed> <= sml::state<statement_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_invalid_request{} ]
      , sml::state<parse_failed> <= sml::state<statement_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_parse_failed{} ]
      , sml::state<parse_failed> <= sml::state<statement_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_internal_error{} ]
      , sml::state<parse_failed> <= sml::state<statement_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_untracked{} ]
      , sml::state<parse_failed> <= sml::state<statement_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_unknown{} ]

      , sml::state<expression_parse_result_decision> <= sml::state<expression_parser::model>
          + sml::completion<event::parse_runtime>

      , sml::state<dispatch_decision> <= sml::state<expression_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_none{} ]

      , sml::state<parse_failed> <= sml::state<expression_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_invalid_request{} ]
      , sml::state<parse_failed> <= sml::state<expression_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_parse_failed{} ]
      , sml::state<parse_failed> <= sml::state<expression_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_internal_error{} ]
      , sml::state<parse_failed> <= sml::state<expression_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_untracked{} ]
      , sml::state<parse_failed> <= sml::state<expression_parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_unknown{} ]

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parse_begin> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<dispatch_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<text_emit> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<comment_emit> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<statement_parse_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<expression_parse_result_decision> + sml::unexpected_event<sml::_>
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

} // namespace emel::text::jinja::parser::program_parser
