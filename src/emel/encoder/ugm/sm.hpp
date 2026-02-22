#pragma once

#include <cstdint>

#include "emel/encoder/ugm/actions.hpp"
#include "emel/encoder/ugm/guards.hpp"
#include "emel/encoder/events.hpp"
#include "emel/sm.hpp"

namespace emel::encoder::ugm {

struct initialized {};
struct encoding {};
struct encode_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * UGM encoder orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting encode intent.
 * - `encoding`/`encode_decision`: run encoder step and branch on phase error.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_encode`/`invalid_encode` validate request pointers and context.
 * - `phase_*` guards observe errors set by actions.
 *
 * action side effects:
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
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encode_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<done> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  sm() : base_type(context_) {}

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  action::context context_{};
};

} // namespace emel::encoder::ugm
