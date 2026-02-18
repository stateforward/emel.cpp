#pragma once

#include <cstdint>

#include "emel/encoder/actions.hpp"
#include "emel/encoder/events.hpp"
#include "emel/encoder/guards.hpp"
#include "emel/sm.hpp"

namespace emel::encoder::bpe {

struct initialized {};
struct encoding {};
struct encode_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * BPE encoder orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting encode intent.
 * - `encoding`/`encode_decision`: run encoder step and branch on phase error.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * Guard semantics:
 * - `valid_encode`/`invalid_encode` validate request pointers and context.
 * - `phase_*` guards observe errors set by actions.
 *
 * Action side effects:
 * - `begin_encode` resets per-request outputs.
 * - `run_encode` performs bounded encoding work.
 * - `mark_done` writes outputs and dispatches done callbacks.
 * - `ensure_last_error` writes errors and dispatches error callbacks.
 * - `on_unexpected` reports sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<initialized> +
            sml::event<event::encode>[guard::valid_encode{}] /
                action::begin_encode = sml::state<encoding>,
        sml::state<initialized> +
            sml::event<event::encode>[guard::invalid_encode{}] /
                action::reject_invalid_encode = sml::state<errored>,

        sml::state<done> + sml::event<event::encode>[guard::valid_encode{}] /
                               action::begin_encode = sml::state<encoding>,
        sml::state<done> + sml::event<event::encode>[guard::invalid_encode{}] /
                               action::reject_invalid_encode =
            sml::state<errored>,

        sml::state<errored> + sml::event<event::encode>[guard::valid_encode{}] /
                                  action::begin_encode = sml::state<encoding>,
        sml::state<errored> +
            sml::event<event::encode>[guard::invalid_encode{}] /
                action::reject_invalid_encode = sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::encode>[guard::valid_encode{}] /
                action::begin_encode = sml::state<encoding>,
        sml::state<unexpected> +
            sml::event<event::encode>[guard::invalid_encode{}] /
                action::reject_invalid_encode = sml::state<unexpected>,

        sml::state<encoding> / action::run_encode = sml::state<encode_decision>,
        sml::state<encode_decision>[guard::phase_ok{}] / action::mark_done =
            sml::state<done>,
        sml::state<encode_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<encoding> +
            sml::event<event::encode> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encode_decision> +
            sml::event<event::encode> / action::on_unexpected =
            sml::state<unexpected>,

        sml::state<initialized> + sml::event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding> + sml::event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encode_decision> +
            sml::event<sml::_> / action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> + sml::event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> + sml::event<sml::_> / action::on_unexpected =
            sml::state<unexpected>);
  }
};

using sm = boost::sml::sm<model>;

} // namespace emel::encoder::bpe
