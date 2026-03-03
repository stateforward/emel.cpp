#pragma once
// benchmark: designed

#include "emel/gbnf/rule_parser/expression_parser/actions.hpp"
#include "emel/gbnf/rule_parser/expression_parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::rule_parser::expression_parser {

struct deciding {};
struct parsed_identifier {};
struct parsed_non_identifier {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<parsed_identifier> <= *sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_identifier{} ]
                 / action::consume_identifier

      , sml::state<parsed_non_identifier> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_non_identifier{} ]
                 / action::consume_non_identifier

      , sml::state<parse_failed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::parse_failed{} ]
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed_identifier>
      , sml::X <= sml::state<parsed_non_identifier>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parsed_identifier> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parsed_non_identifier> + sml::unexpected_event<sml::_>
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

}  // namespace emel::gbnf::rule_parser::expression_parser
