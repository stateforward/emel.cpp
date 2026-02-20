#pragma once

#include <cstdint>

#include "emel/encoder/plamo2/actions.hpp"
#include "emel/encoder/plamo2/guards.hpp"
#include "emel/encoder/events.hpp"
#include "emel/sm.hpp"

namespace emel::encoder::plamo2 {

struct initialized {};
struct encoding {};
struct encode_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * PLaMo2 encoder orchestration model.
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
        sml::state<initialized> +
            sml::event<events::encoding_done> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<initialized> +
            sml::event<events::encoding_error> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding> +
            sml::event<events::encoding_done> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding> +
            sml::event<events::encoding_error> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encode_decision> +
            sml::event<events::encoding_done> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encode_decision> +
            sml::event<events::encoding_error> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<done> +
            sml::event<events::encoding_done> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<done> +
            sml::event<events::encoding_error> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> +
            sml::event<events::encoding_done> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> +
            sml::event<events::encoding_error> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<events::encoding_done> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<events::encoding_error> / action::on_unexpected =
            sml::state<unexpected>,

        sml::state<initialized> +
            sml::event<sml::_>[guard::not_internal_event{}] / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding> +
            sml::event<sml::_>[guard::not_internal_event{}] / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encode_decision> +
            sml::event<sml::_>[guard::not_internal_event{}] / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<done> +
            sml::event<sml::_>[guard::not_internal_event{}] / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> +
            sml::event<sml::_>[guard::not_internal_event{}] / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<sml::_>[guard::not_internal_event{}] / action::on_unexpected =
            sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  using base_type::base_type;
  using base_type::process_event;
};

} // namespace emel::encoder::plamo2
