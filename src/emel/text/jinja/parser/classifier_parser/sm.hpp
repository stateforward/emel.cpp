#pragma once

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/classifier_parser/actions.hpp"
#include "emel/text/jinja/parser/classifier_parser/guards.hpp"
#include "emel/text/jinja/parser/events.hpp"

namespace emel::text::jinja::parser::classifier_parser {

struct deciding {};
struct statement_decision {};
struct expression_decision {};
struct classification_result_decision {};
struct done {};
struct errored {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Classification setup.
        sml::state<statement_decision> <= *sml::state<deciding>
          + sml::completion<event::parse_runtime>
          / action::begin_classification

      //------------------------------------------------------------------------------//
      // Statement classifier.
      , sml::state<classification_result_decision> <= sml::state<statement_decision>
          + sml::completion<event::parse_runtime>[ guard::no_tokens{} ]
          / action::set_statement_unknown

      , sml::state<classification_result_decision> <= sml::state<statement_decision>
          + sml::completion<event::parse_runtime>[ guard::token_text{} ]
          / action::set_statement_text

      , sml::state<classification_result_decision> <= sml::state<statement_decision>
          + sml::completion<event::parse_runtime>[ guard::token_comment{} ]
          / action::set_statement_comment

      , sml::state<expression_decision> <= sml::state<statement_decision>
          + sml::completion<event::parse_runtime>[ guard::token_open_expression{} ]
          / action::set_statement_expression

      , sml::state<classification_result_decision> <= sml::state<statement_decision>
          + sml::completion<event::parse_runtime>[ guard::token_open_statement{} ]
          / action::set_statement_statement

      , sml::state<classification_result_decision> <= sml::state<statement_decision>
          + sml::completion<event::parse_runtime>[ guard::token_unknown{} ]
          / action::set_statement_unknown

      //------------------------------------------------------------------------------//
      // Expression classifier.
      , sml::state<classification_result_decision> <= sml::state<expression_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_no_token{} ]
          / action::set_expression_unknown

      , sml::state<classification_result_decision> <= sml::state<expression_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_token_literal{} ]
          / action::set_expression_literal

      , sml::state<classification_result_decision> <= sml::state<expression_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_token_identifier{} ]
          / action::set_expression_identifier

      , sml::state<classification_result_decision> <= sml::state<expression_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_token_unary{} ]
          / action::set_expression_unary

      , sml::state<classification_result_decision> <= sml::state<expression_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_token_compound{} ]
          / action::set_expression_compound

      , sml::state<classification_result_decision> <= sml::state<expression_decision>
          + sml::completion<event::parse_runtime>[ guard::expr_token_unknown{} ]
          / action::set_expression_unknown

      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<classification_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_none{} ]
      , sml::state<errored> <= sml::state<classification_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_invalid_request{} ]
      , sml::state<errored> <= sml::state<classification_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_parse_failed{} ]
      , sml::state<errored> <= sml::state<classification_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_internal_error{} ]
      , sml::state<errored> <= sml::state<classification_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_untracked{} ]
      , sml::state<errored> <= sml::state<classification_result_decision>
          + sml::completion<event::parse_runtime>[ guard::parse_error_unknown{} ]

      , sml::X <= sml::state<done>
      , sml::X <= sml::state<errored>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<statement_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<expression_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<classification_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<errored> + sml::unexpected_event<sml::_>
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

}  // namespace emel::text::jinja::parser::classifier_parser
