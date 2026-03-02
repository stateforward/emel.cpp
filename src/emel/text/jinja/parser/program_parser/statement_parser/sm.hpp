#pragma once

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
#include "emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"

namespace emel::text::jinja::parser::program_parser::statement_parser {

struct deciding {};
struct statement_kind_decision {};
struct statement_scan {};
struct parsed {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Statement parser.
        sml::state<statement_kind_decision> <= *sml::state<deciding>
          + sml::completion<event::parse_runtime>

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_set{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_if{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_elif{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_else{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endif{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_for{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endfor{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_macro{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endmacro{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_call{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endcall{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_filter{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endfilter{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_break{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_continue{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_generation{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endgeneration{} ]
          / action::begin_statement_scan

      , sml::state<statement_scan> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_endset{} ]
          / action::begin_statement_scan

      , sml::state<parse_failed> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_identifier_missing{} ]
          / action::fail_statement_open_token

      , sml::state<parse_failed> <= sml::state<statement_kind_decision>
          + sml::completion<event::parse_runtime>[ guard::statement_name_unknown{} ]
          / action::fail_statement_name_token

      , sml::state<parsed> <= sml::state<statement_scan>
          + sml::completion<event::parse_runtime>[ guard::statement_scan_at_close{} ]
          / action::consume_statement_close_and_emit

      , sml::state<statement_scan> <= sml::state<statement_scan>
          + sml::completion<event::parse_runtime>[ guard::statement_scan_continue{} ]
          / action::consume_statement_token

      , sml::state<parse_failed> <= sml::state<statement_scan>
          + sml::completion<event::parse_runtime>[ guard::statement_scan_eof{} ]
          / action::fail_statement_start_token

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<statement_kind_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<statement_scan> + sml::unexpected_event<sml::_>
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

} // namespace emel::text::jinja::parser::program_parser::statement_parser
