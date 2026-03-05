#pragma once
// benchmark: designed

#include "emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
#include "emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::rule_parser::nonterm_parser {

struct deciding {};
struct definition_lookup_exec {};
struct definition_lookup_decision {};
struct reference_lookup_exec {};
struct reference_lookup_decision {};
struct parsed {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<definition_lookup_exec> <= *sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_identifier_definition{} ]

      , sml::state<reference_lookup_exec> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_identifier_reference{} ]

      , sml::state<parse_failed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::state<definition_lookup_decision> <= sml::state<definition_lookup_exec> + sml::completion<rule_parser::event::parse_rules>
                 / action::lookup_definition_candidate

      , sml::state<reference_lookup_decision> <= sml::state<reference_lookup_exec> + sml::completion<rule_parser::event::parse_rules>
                 / action::lookup_reference_candidate

      //------------------------------------------------------------------------------//
      , sml::state<parsed> <= sml::state<definition_lookup_decision> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::definition_existing_valid{} ]
                 / action::consume_definition_existing

      , sml::state<parsed> <= sml::state<definition_lookup_decision> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::definition_new_valid{} ]
                 / action::consume_definition_new

      , sml::state<parse_failed> <= sml::state<definition_lookup_decision> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::definition_failed{} ]
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::state<parsed> <= sml::state<reference_lookup_decision> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::reference_existing_valid{} ]
                 / action::consume_reference_existing

      , sml::state<parsed> <= sml::state<reference_lookup_decision> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::reference_new_valid{} ]
                 / action::consume_reference_new

      , sml::state<parse_failed> <= sml::state<reference_lookup_decision> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::reference_failed{} ]
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<definition_lookup_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<definition_lookup_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<reference_lookup_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<reference_lookup_decision> + sml::unexpected_event<sml::_>
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

}  // namespace emel::gbnf::rule_parser::nonterm_parser
