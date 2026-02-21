#pragma once

#include <cstdint>

#include "emel/gbnf/parser/actions.hpp"
#include "emel/gbnf/parser/events.hpp"
#include "emel/gbnf/parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::parser {

struct initialized {};
struct parse_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * GBNF parser orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting parse intent.
 * - `parse_decision`: run parsing step and branch based on phase
 * results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_parse`/`invalid_parse` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * action side effects:
 * - `run_parse` parses grammar synchronously and dispatches callbacks.
 * - `on_unexpected` reports any event sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        // 1. start parsing
        *sml::state<initialized> +
            sml::event<event::parse>[guard::valid_parse{}] /
                action::run_parse = sml::state<parse_decision>,
        sml::state<initialized> +
            sml::event<event::parse>[guard::invalid_parse{}] /
                action::reject_invalid_parse = sml::state<errored>,

        sml::state<done> + sml::event<event::parse>[guard::valid_parse{}] /
                               action::run_parse = sml::state<parse_decision>,
        sml::state<done> + sml::event<event::parse>[guard::invalid_parse{}] /
                               action::reject_invalid_parse =
            sml::state<errored>,

        sml::state<errored> + sml::event<event::parse>[guard::valid_parse{}] /
                                  action::run_parse = sml::state<parse_decision>,
        sml::state<errored> + sml::event<event::parse>[guard::invalid_parse{}] /
                                  action::reject_invalid_parse =
            sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::parse>[guard::valid_parse{}] /
                action::run_parse = sml::state<parse_decision>,
        sml::state<unexpected> +
            sml::event<event::parse>[guard::invalid_parse{}] /
                action::reject_invalid_parse = sml::state<unexpected>,

        sml::state<parse_decision>[guard::phase_ok{}] =
            sml::state<done>,
        sml::state<parse_decision>[guard::phase_failed{}] =
            sml::state<errored>,

        sml::state<initialized> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<parse_decision> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
                               action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
                                  action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  using base_type::base_type;
  using base_type::process_event;
};

} // namespace emel::gbnf::parser
