#pragma once
// benchmark: designed

#include <cstdint>

#include "emel/gbnf/rule_parser/definition_parser/sm.hpp"
#include "emel/gbnf/rule_parser/expression_parser/sm.hpp"
#include "emel/gbnf/rule_parser/nonterm_parser/sm.hpp"
#include "emel/gbnf/rule_parser/actions.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/guards.hpp"
#include "emel/gbnf/rule_parser/term_parser/sm.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::rule_parser {

struct ready {};

struct expect_rule_name {};
struct expect_rule_name_decision {};

struct expect_definition {};
struct expect_definition_decision {};

struct in_rule_expression_need_term {};
struct in_rule_expression_need_term_decision {};

struct in_rule_expression_after_term {};
struct in_rule_expression_after_term_decision {};
struct rule_reference_decision {};
struct rule_reference_parse_exec {};
struct rule_reference_shape_decision {};
struct rule_reference_parse_result_decision {};
struct quantifier_decision {};
struct quantifier_parse_exec {};
struct quantifier_shape_decision {};
struct quantifier_braced_shape_decision {};
struct quantifier_braced_exact_parse_exec {};
struct quantifier_braced_open_parse_exec {};
struct quantifier_braced_range_parse_exec {};
struct quantifier_parse_result_decision {};

struct eof_symbols_decision {};
struct parse_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<expect_rule_name> <= *sml::state<ready> + sml::event<event::parse_rules>
                 [ guard::valid_parse{} ]
                 / action::begin_parse

      , sml::state<ready> <= sml::state<ready> + sml::event<event::parse_rules>
                 [ guard::invalid_parse_with_dispatchable_grammar{} ]
                 / action::reject_invalid_parse_with_dispatch

      , sml::state<ready> <= sml::state<ready> + sml::event<event::parse_rules>
                 [ guard::invalid_parse_with_grammar_only{} ]
                 / action::reject_invalid_parse_with_grammar_only

      , sml::state<ready> <= sml::state<ready> + sml::event<event::parse_rules>
                 [ guard::invalid_parse_without_grammar{} ]
                 / action::reject_invalid_parse_without_grammar

      //------------------------------------------------------------------------------//
      // Rule name phase.
      , sml::state<expect_rule_name_decision> <= sml::state<expect_rule_name> + sml::completion<event::parse_rules>
                 / action::request_next_token

      , sml::state<parse_decision> <= sml::state<expect_rule_name_decision> + sml::completion<event::parse_rules>
                 [ guard::lexer_failed{} ]

      , sml::state<eof_symbols_decision> <= sml::state<expect_rule_name_decision> + sml::completion<event::parse_rules>
                 [ guard::lexer_at_eof{} ]

      , sml::state<nonterm_parser::model> <= sml::state<expect_rule_name_decision> + sml::completion<event::parse_rules>
                 [ guard::token_identifier{} ]
                 / action::set_nonterm_mode_definition

      , sml::state<expect_rule_name> <= sml::state<expect_rule_name_decision> + sml::completion<event::parse_rules>
                 [ guard::token_newline{} ]

      , sml::state<parse_decision> <= sml::state<expect_rule_name_decision> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Nonterminal classifier.
      , sml::state<parse_decision> <= sml::state<nonterm_parser::model> + sml::completion<event::parse_rules>
                 [ guard::nonterm_failed{} ]

      , sml::state<expect_definition> <= sml::state<nonterm_parser::model> + sml::completion<event::parse_rules>
                 [ guard::nonterm_definition_done{} ]
                 / action::apply_nonterm_definition

      , sml::state<in_rule_expression_after_term> <= sml::state<nonterm_parser::model> + sml::completion<event::parse_rules>
                 [ guard::nonterm_reference_done{} ]
                 / action::apply_nonterm_reference

      , sml::state<parse_decision> <= sml::state<nonterm_parser::model> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Definition operator phase.
      , sml::state<expect_definition_decision> <= sml::state<expect_definition> + sml::completion<event::parse_rules>
                 / action::request_next_token

      , sml::state<parse_decision> <= sml::state<expect_definition_decision> + sml::completion<event::parse_rules>
                 [ guard::lexer_failed{} ]

      , sml::state<parse_decision> <= sml::state<expect_definition_decision> + sml::completion<event::parse_rules>
                 [ guard::lexer_at_eof{} ]
                 / action::fail_eof_in_expect_definition

      , sml::state<definition_parser::model> <= sml::state<expect_definition_decision> + sml::completion<event::parse_rules>
                 [ guard::lexer_has_token{} ]

      , sml::state<parse_decision> <= sml::state<expect_definition_decision> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      , sml::state<parse_decision> <= sml::state<definition_parser::model> + sml::completion<event::parse_rules>
                 [ guard::definition_failed{} ]

      , sml::state<in_rule_expression_need_term> <= sml::state<definition_parser::model> + sml::completion<event::parse_rules>
                 [ guard::definition_done{} ]
                 / action::consume_token_definition_operator

      , sml::state<parse_decision> <= sml::state<definition_parser::model> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Expression phase: need term.
      , sml::state<in_rule_expression_need_term_decision> <= sml::state<in_rule_expression_need_term> + sml::completion<event::parse_rules>
                 / action::request_next_token

      , sml::state<parse_decision> <= sml::state<in_rule_expression_need_term_decision> +
               sml::completion<event::parse_rules>
                 [ guard::lexer_failed{} ]

      , sml::state<expression_parser::model> <= sml::state<in_rule_expression_need_term_decision> +
               sml::completion<event::parse_rules>
                 [ guard::lexer_has_token{} ]
                 / action::set_term_origin_need_term

      , sml::state<parse_decision> <= sml::state<in_rule_expression_need_term_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Expression phase: after term.
      , sml::state<in_rule_expression_after_term_decision> <= sml::state<in_rule_expression_after_term> + sml::completion<event::parse_rules>
                 / action::request_next_token

      , sml::state<parse_decision> <= sml::state<in_rule_expression_after_term_decision> +
               sml::completion<event::parse_rules>
                 [ guard::lexer_failed{} ]

      , sml::state<eof_symbols_decision> <= sml::state<in_rule_expression_after_term_decision> +
               sml::completion<event::parse_rules>
                 [ guard::eof_can_finalize_active_rule{} ]
                 / action::finalize_active_rule_on_eof

      , sml::state<parse_decision> <= sml::state<in_rule_expression_after_term_decision> +
               sml::completion<event::parse_rules>
                 [ guard::eof_cannot_finalize_active_rule{} ]
                 / action::consume_token_invalid

      , sml::state<expression_parser::model> <= sml::state<in_rule_expression_after_term_decision> +
               sml::completion<event::parse_rules>
                 [ guard::lexer_has_token{} ]
                 / action::set_term_origin_after_term

      , sml::state<parse_decision> <= sml::state<in_rule_expression_after_term_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Expression classifier.
      , sml::state<parse_decision> <= sml::state<expression_parser::model> + sml::completion<event::parse_rules>
                 [ guard::expression_failed{} ]

      , sml::state<nonterm_parser::model> <= sml::state<expression_parser::model> + sml::completion<event::parse_rules>
                 [ guard::expression_done_identifier{} ]
                 / action::set_nonterm_mode_reference

      , sml::state<term_parser::model> <= sml::state<expression_parser::model> + sml::completion<event::parse_rules>
                 [ guard::expression_done_non_identifier{} ]

      , sml::state<parse_decision> <= sml::state<expression_parser::model> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Term classifier and reducers.
      , sml::state<parse_decision> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_failed{} ]

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_need_literal_valid{} ]
                 / action::consume_token_literal

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_need_character_class_valid{} ]
                 / action::consume_token_character_class

      , sml::state<rule_reference_decision> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_need_rule_reference_candidate{} ]

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_need_dot_valid{} ]
                 / action::consume_token_dot

      , sml::state<in_rule_expression_need_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_need_open_group_valid{} ]
                 / action::consume_token_open_group

      , sml::state<in_rule_expression_need_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_need_newline_with_group_depth_nonzero{} ]

      , sml::state<parse_decision> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_from_need_term{} ]
                 / action::consume_token_invalid

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_literal_valid{} ]
                 / action::consume_token_literal

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_character_class_valid{} ]
                 / action::consume_token_character_class

      , sml::state<rule_reference_decision> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_rule_reference_candidate{} ]

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_dot_valid{} ]
                 / action::consume_token_dot

      , sml::state<in_rule_expression_need_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_open_group_valid{} ]
                 / action::consume_token_open_group

      , sml::state<in_rule_expression_need_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_alternation_valid{} ]
                 / action::consume_token_alternation

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_newline_with_group_depth_nonzero{} ]

      , sml::state<expect_rule_name> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_newline_with_group_depth_zero_valid{} ]
                 / action::finalize_active_rule_on_eof

      , sml::state<in_rule_expression_after_term> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_close_group_valid{} ]
                 / action::consume_token_close_group

      , sml::state<quantifier_decision> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_after_quantifier_candidate{} ]

      , sml::state<parse_decision> <= sml::state<term_parser::model> + sml::completion<event::parse_rules>
                 [ guard::term_from_after_term{} ]
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Rule reference classifier.
      , sml::state<rule_reference_parse_exec> <= sml::state<rule_reference_decision> +
               sml::completion<event::parse_rules>
                 [ guard::token_rule_reference_candidate{} ]
                 / action::prepare_rule_reference_parse

      , sml::state<parse_decision> <= sml::state<rule_reference_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      , sml::state<rule_reference_shape_decision> <=
               sml::state<rule_reference_parse_exec> + sml::completion<event::parse_rules>
      , sml::state<rule_reference_parse_result_decision> <=
               sml::state<rule_reference_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::rule_reference_plain_envelope_valid{} ]
                 / action::parse_rule_reference_plain_candidate
      , sml::state<rule_reference_parse_result_decision> <=
               sml::state<rule_reference_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::rule_reference_negated_envelope_valid{} ]
                 / action::parse_rule_reference_negated_candidate
      , sml::state<parse_decision> <=
               sml::state<rule_reference_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::rule_reference_plain_envelope_invalid{} ]
                 / action::consume_token_invalid
      , sml::state<parse_decision> <=
               sml::state<rule_reference_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::rule_reference_negated_envelope_invalid{} ]
                 / action::consume_token_invalid
      , sml::state<parse_decision> <=
               sml::state<rule_reference_shape_decision> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      , sml::state<in_rule_expression_after_term> <=
               sml::state<rule_reference_parse_result_decision> +
               sml::completion<event::parse_rules>
                 [ guard::parsed_rule_reference_plain_valid{} ]
                 / action::consume_token_rule_reference_plain_parsed

      , sml::state<in_rule_expression_after_term> <=
               sml::state<rule_reference_parse_result_decision> +
               sml::completion<event::parse_rules>
                 [ guard::parsed_rule_reference_negated_valid{} ]
                 / action::consume_token_rule_reference_negated_parsed

      , sml::state<parse_decision> <=
               sml::state<rule_reference_parse_result_decision> +
               sml::completion<event::parse_rules>
                 [ guard::parsed_rule_reference_invalid{} ]
                 / action::consume_token_invalid

      , sml::state<parse_decision> <=
               sml::state<rule_reference_parse_result_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Quantifier classifier.
      , sml::state<quantifier_parse_exec> <= sml::state<quantifier_decision> +
               sml::completion<event::parse_rules>
                 [ guard::quantifier_candidate{} ]
                 / action::prepare_quantifier_parse

      , sml::state<parse_decision> <= sml::state<quantifier_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      , sml::state<quantifier_shape_decision> <=
               sml::state<quantifier_parse_exec> + sml::completion<event::parse_rules>
      , sml::state<quantifier_parse_result_decision> <=
               sml::state<quantifier_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::quantifier_token_star{} ]
                 / action::parse_quantifier_star_bounds_candidate
      , sml::state<quantifier_parse_result_decision> <=
               sml::state<quantifier_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::quantifier_token_plus{} ]
                 / action::parse_quantifier_plus_bounds_candidate
      , sml::state<quantifier_parse_result_decision> <=
               sml::state<quantifier_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::quantifier_token_question{} ]
                 / action::parse_quantifier_question_bounds_candidate
      , sml::state<quantifier_braced_shape_decision> <=
               sml::state<quantifier_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::quantifier_token_braced{} ]
      , sml::state<quantifier_braced_exact_parse_exec> <=
               sml::state<quantifier_braced_shape_decision> +
               sml::completion<event::parse_rules>
                 [ guard::quantifier_braced_exact_shape{} ]
      , sml::state<quantifier_braced_open_parse_exec> <=
               sml::state<quantifier_braced_shape_decision> +
               sml::completion<event::parse_rules>
                 [ guard::quantifier_braced_open_shape{} ]
      , sml::state<quantifier_braced_range_parse_exec> <=
               sml::state<quantifier_braced_shape_decision> +
               sml::completion<event::parse_rules>
                 [ guard::quantifier_braced_range_shape{} ]
      , sml::state<quantifier_parse_result_decision> <=
               sml::state<quantifier_braced_exact_parse_exec> +
               sml::completion<event::parse_rules>
                 / action::parse_quantifier_braced_exact_bounds_candidate
      , sml::state<quantifier_parse_result_decision> <=
               sml::state<quantifier_braced_open_parse_exec> +
               sml::completion<event::parse_rules>
                 / action::parse_quantifier_braced_open_bounds_candidate
      , sml::state<quantifier_parse_result_decision> <=
               sml::state<quantifier_braced_range_parse_exec> +
               sml::completion<event::parse_rules>
                 / action::parse_quantifier_braced_range_bounds_candidate
      , sml::state<parse_decision> <=
               sml::state<quantifier_braced_shape_decision> +
               sml::completion<event::parse_rules>
                 [ guard::quantifier_braced_invalid_shape{} ]
                 / action::consume_token_invalid
      , sml::state<parse_decision> <=
               sml::state<quantifier_braced_shape_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid
      , sml::state<parse_decision> <=
               sml::state<quantifier_shape_decision> + sml::completion<event::parse_rules>
                 [ guard::quantifier_token_unknown{} ]
                 / action::consume_token_invalid
      , sml::state<parse_decision> <=
               sml::state<quantifier_shape_decision> + sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      , sml::state<in_rule_expression_after_term> <=
               sml::state<quantifier_parse_result_decision> +
               sml::completion<event::parse_rules>
                 [ guard::parsed_quantifier_applicable{} ]
                 / action::consume_token_quantifier_parsed

      , sml::state<parse_decision> <=
               sml::state<quantifier_parse_result_decision> +
               sml::completion<event::parse_rules>
                 [ guard::parsed_quantifier_invalid{} ]
                 / action::consume_token_invalid

      , sml::state<parse_decision> <=
               sml::state<quantifier_parse_result_decision> +
               sml::completion<event::parse_rules>
                 [ guard::parsed_quantifier_not_applicable{} ]
                 / action::consume_token_invalid

      , sml::state<parse_decision> <=
               sml::state<quantifier_parse_result_decision> +
               sml::completion<event::parse_rules>
                 / action::consume_token_invalid

      //------------------------------------------------------------------------------//
      // Finalization and outcome dispatch.
      , sml::state<parse_decision> <= sml::state<eof_symbols_decision> + sml::completion<event::parse_rules>
                 [ guard::eof_can_finalize_symbols{} ]

      , sml::state<parse_decision> <= sml::state<eof_symbols_decision> + sml::completion<event::parse_rules>
                 [ guard::eof_cannot_finalize_symbols{} ]
                 / action::consume_token_invalid

      , sml::state<ready> <= sml::state<parse_decision> + sml::completion<event::parse_rules>
                 [ guard::phase_ok{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<parse_decision> + sml::completion<event::parse_rules>
                 [ guard::phase_failed{} ]
                 / action::dispatch_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<expect_rule_name> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<expect_rule_name_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<expect_definition> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<expect_definition_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<in_rule_expression_need_term> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<in_rule_expression_need_term_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<in_rule_expression_after_term> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<in_rule_expression_after_term_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<rule_reference_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<rule_reference_parse_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<rule_reference_shape_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<rule_reference_parse_result_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_parse_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_shape_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_braced_shape_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_braced_exact_parse_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_braced_open_parse_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_braced_range_parse_exec> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<quantifier_parse_result_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<parse_decision> <= sml::state<eof_symbols_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<parse_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;
  using base_type::process_event;

  bool process_event(const event::parse & ev) {
    event::parse_rules_ctx ctx{};
    event::parse_rules evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::gbnf::rule_parser
