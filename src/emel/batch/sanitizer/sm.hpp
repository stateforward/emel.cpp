#pragma once

#include <cstdint>

#include "emel/batch/sanitizer/actions.hpp"
#include "emel/batch/sanitizer/events.hpp"
#include "emel/batch/sanitizer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::batch::sanitizer {

struct initialized {};
struct sanitizing {};
struct sanitize_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * batch sanitizer orchestration model (decode-only).
 *
 * state purposes:
 * - `initialized`: idle state awaiting sanitize intent.
 * - `sanitizing`/`sanitize_decision`: run sanitizer logic and branch on result.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_request`/`invalid_request`: validate request pointers and capacity.
 * - `phase_ok`/`phase_failed`: observe errors set by actions.
 *
 * action side effects:
 * - `begin_sanitize`: capture inputs and reset outputs.
 * - `run_sanitize_decode`: validate and normalize batch fields.
 * - `mark_done`: clear error state.
 * - `ensure_last_error`: ensure a terminal error code.
 * - `on_unexpected`: report sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::sanitize_decode>[guard::valid_request{}] /
          action::begin_sanitize = sml::state<sanitizing>,
      sml::state<initialized> + sml::event<event::sanitize_decode>[guard::invalid_request{}] /
          action::reject_invalid_sanitize = sml::state<errored>,

      sml::state<sanitizing> / action::run_sanitize_decode = sml::state<sanitize_decision>,
      sml::state<sanitize_decision>[guard::phase_failed{}] = sml::state<errored>,
      sml::state<sanitize_decision>[guard::phase_ok{}] / action::mark_done =
          sml::state<done>,

      sml::state<done> + sml::event<event::sanitize_decode>[guard::valid_request{}] /
          action::begin_sanitize = sml::state<sanitizing>,
      sml::state<done> + sml::event<event::sanitize_decode>[guard::invalid_request{}] /
          action::reject_invalid_sanitize = sml::state<errored>,

      sml::state<errored> + sml::event<event::sanitize_decode>[guard::valid_request{}] /
          action::begin_sanitize = sml::state<sanitizing>,
      sml::state<errored> + sml::event<event::sanitize_decode>[guard::invalid_request{}] /
          action::reject_invalid_sanitize = sml::state<errored>,

      sml::state<unexpected> + sml::event<event::sanitize_decode>[guard::valid_request{}] /
          action::begin_sanitize = sml::state<sanitizing>,
      sml::state<unexpected> + sml::event<event::sanitize_decode>[guard::invalid_request{}] /
          action::reject_invalid_sanitize = sml::state<errored>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<sanitizing> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<sanitize_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<done> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<unexpected> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  sm() : base_type(context_) {}
  using base_type::process_event;

 private:
  action::context context_{};
};

}  // namespace emel::batch::sanitizer
