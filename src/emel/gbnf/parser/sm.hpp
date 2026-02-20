#pragma once

#include <cstdint>

#include "emel/gbnf/parser/actions.hpp"
#include "emel/gbnf/parser/events.hpp"
#include "emel/gbnf/parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::parser {

struct Initialized {};
struct ParseDecision {};
struct Done {};
struct Errored {};
struct Unexpected {};

/**
 * GBNF parser orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting parse intent.
 * - `parse_decision`: run parsing step and branch based on phase
 * results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * Guard semantics:
 * - `valid_parse`/`invalid_parse` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * Action side effects:
 * - `run_parse` parses grammar synchronously and dispatches callbacks.
 * - `on_unexpected` reports any event sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        // 1. Start parsing
        *sml::state<Initialized> +
            sml::event<event::parse>[guard::valid_parse{}] /
                action::run_parse = sml::state<ParseDecision>,
        sml::state<Initialized> +
            sml::event<event::parse>[guard::invalid_parse{}] /
                action::reject_invalid_parse = sml::state<Errored>,

        sml::state<Done> + sml::event<event::parse>[guard::valid_parse{}] /
                               action::run_parse = sml::state<ParseDecision>,
        sml::state<Done> + sml::event<event::parse>[guard::invalid_parse{}] /
                               action::reject_invalid_parse =
            sml::state<Errored>,

        sml::state<Errored> + sml::event<event::parse>[guard::valid_parse{}] /
                                  action::run_parse = sml::state<ParseDecision>,
        sml::state<Errored> + sml::event<event::parse>[guard::invalid_parse{}] /
                                  action::reject_invalid_parse =
            sml::state<Errored>,

        sml::state<Unexpected> +
            sml::event<event::parse>[guard::valid_parse{}] /
                action::run_parse = sml::state<ParseDecision>,
        sml::state<Unexpected> +
            sml::event<event::parse>[guard::invalid_parse{}] /
                action::reject_invalid_parse = sml::state<Unexpected>,

        sml::state<ParseDecision>[guard::phase_ok{}] =
            sml::state<Done>,
        sml::state<ParseDecision>[guard::phase_failed{}] =
            sml::state<Errored>,

        sml::state<Initialized> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<Unexpected>,
        sml::state<ParseDecision> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<Unexpected>,
        sml::state<Done> + sml::unexpected_event<sml::_> /
                               action::on_unexpected = sml::state<Unexpected>,
        sml::state<Errored> + sml::unexpected_event<sml::_> /
                                  action::on_unexpected =
            sml::state<Unexpected>,
        sml::state<Unexpected> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<Unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  using base_type::base_type;
  using base_type::process_event;
};

} // namespace emel::gbnf::parser
