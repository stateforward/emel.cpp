#pragma once
// benchmark: designed

#include "emel/gbnf/rule_parser/term_parser/actions.hpp"
#include "emel/gbnf/rule_parser/term_parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::rule_parser::term_parser {

struct deciding {};
struct parsed {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<parsed> <= *sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_string_literal{} ]
                 / action::consume_string_literal

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_character_class{} ]
                 / action::consume_character_class

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_rule_reference{} ]
                 / action::consume_rule_reference

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_dot{} ]
                 / action::consume_dot

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_open_group{} ]
                 / action::consume_open_group

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_close_group{} ]
                 / action::consume_close_group

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_quantifier{} ]
                 / action::consume_quantifier

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_alternation{} ]
                 / action::consume_alternation

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::token_newline{} ]
                 / action::consume_newline

      , sml::state<parse_failed> <= sml::state<deciding> + sml::completion<rule_parser::event::parse_rules>
                 [ guard::parse_failed{} ]
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
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

}  // namespace emel::gbnf::rule_parser::term_parser
