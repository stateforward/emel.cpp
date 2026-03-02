#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/classifier_parser/sm.hpp"
#include "emel/text/jinja/parser/context.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/guards.hpp"
#include "emel/text/jinja/parser/lexer/sm.hpp"
#include "emel/text/jinja/parser/program_parser/sm.hpp"

namespace emel::text::jinja::parser {

struct initialized {};
struct request_decision {};
struct tokenize_begin {};
struct tokenize_next {};
struct tokenize_result_decision {};
struct tokenize_append {};
struct classify_result_decision {};
struct parse_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<request_decision> <= *sml::state<initialized>
          + sml::event<event::parse_runtime>[ guard::valid_parse{} ]
          / action::begin_parse
      , sml::state<parse_result_decision> <= sml::state<initialized>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_with_callbacks{} ]
          / action::reject_invalid_parse
      , sml::state<errored> <= sml::state<initialized>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_without_callbacks{} ]
          / action::reject_invalid_parse

      , sml::state<request_decision> <= sml::state<done>
          + sml::event<event::parse_runtime>[ guard::valid_parse{} ]
          / action::begin_parse
      , sml::state<parse_result_decision> <= sml::state<done>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_with_callbacks{} ]
          / action::reject_invalid_parse
      , sml::state<errored> <= sml::state<done>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_without_callbacks{} ]
          / action::reject_invalid_parse

      , sml::state<request_decision> <= sml::state<errored>
          + sml::event<event::parse_runtime>[ guard::valid_parse{} ]
          / action::begin_parse
      , sml::state<parse_result_decision> <= sml::state<errored>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_with_callbacks{} ]
          / action::reject_invalid_parse
      , sml::state<errored> <= sml::state<errored>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_without_callbacks{} ]
          / action::reject_invalid_parse

      , sml::state<request_decision> <= sml::state<unexpected>
          + sml::event<event::parse_runtime>[ guard::valid_parse{} ]
          / action::begin_parse
      , sml::state<parse_result_decision> <= sml::state<unexpected>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_with_callbacks{} ]
          / action::reject_invalid_parse
      , sml::state<errored> <= sml::state<unexpected>
          + sml::event<event::parse_runtime>[ guard::invalid_parse_without_callbacks{} ]
          / action::reject_invalid_parse

      //------------------------------------------------------------------------------//
      // Pipeline phases.
      , sml::state<tokenize_begin> <= sml::state<request_decision>
          + sml::completion<event::parse_runtime>
          / action::begin_tokenization

      , sml::state<tokenize_next> <= sml::state<tokenize_begin>
          + sml::completion<event::parse_runtime>
          / action::request_next_lex_token

      , sml::state<tokenize_result_decision> <= sml::state<tokenize_next>
          + sml::completion<event::parse_runtime>

      , sml::state<classifier_parser::model> <= sml::state<tokenize_result_decision>
          + sml::completion<event::parse_runtime>[ guard::lexer_at_eof{} ]

      , sml::state<tokenize_append> <= sml::state<tokenize_result_decision>
          + sml::completion<event::parse_runtime>[ guard::lexer_has_token{} ]
          / action::append_lex_token

      , sml::state<parse_result_decision> <= sml::state<tokenize_result_decision>
          + sml::completion<event::parse_runtime>[ guard::phase_failed{} ]
          / action::commit_lex_error

      , sml::state<tokenize_next> <= sml::state<tokenize_append>
          + sml::completion<event::parse_runtime>
          / action::request_next_lex_token

      , sml::state<classify_result_decision> <= sml::state<classifier_parser::model>
          + sml::completion<event::parse_runtime>

      , sml::state<program_parser::model> <= sml::state<classify_result_decision>
          + sml::completion<event::parse_runtime>[ guard::phase_ok{} ]
      , sml::state<parse_result_decision> <= sml::state<classify_result_decision>
          + sml::completion<event::parse_runtime>[ guard::phase_failed{} ]

      , sml::state<parse_result_decision> <= sml::state<program_parser::model>
          + sml::completion<event::parse_runtime>

      , sml::state<done> <= sml::state<parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::phase_ok{} ]
          / action::dispatch_done
      , sml::state<errored> <= sml::state<parse_result_decision>
          + sml::completion<event::parse_runtime>[ guard::phase_failed{} ]
          / action::dispatch_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<unexpected> <= sml::state<initialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<tokenize_begin> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<tokenize_next> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<tokenize_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<tokenize_append> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<classify_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<parse_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::parse &ev) {
    event::parse_ctx runtime_ctx{
        ev.template_text,
        ev.error_out,
        ev.error_pos_out,
    };
    event::parse_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && runtime_ctx.err == error::none;
  }
};

using Parser = sm;

} // namespace emel::text::jinja::parser
