#pragma once

#include <cstdint>

#include "emel/token/batcher/actions.hpp"
#include "emel/token/batcher/events.hpp"
#include "emel/token/batcher/guards.hpp"
#include "emel/sm.hpp"

namespace emel::token::batcher {

struct initialized {};
struct batching {};
struct batch_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * batch normalization orchestration model (decode-only).
 *
 * state purposes:
 * - `initialized`: idle state awaiting batch intent.
 * - `batching`/`batch_decision`: run batch normalization logic and branch on result.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_request`/`invalid_request`: validate request pointers and capacity.
 * - `phase_ok`/`phase_failed`: observe errors set by actions.
 *
 * action side effects:
 * - `begin_batch`: capture inputs and reset outputs.
 * - `run_batch`: validate and normalize batch fields.
 * - `mark_done`: clear error state.
 * - `ensure_last_error`: ensure a terminal error code.
 * - `on_unexpected`: report sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::batch>[guard::valid_request{}] /
          action::begin_batch = sml::state<batching>,
      sml::state<initialized> + sml::event<event::batch>[guard::invalid_request{}] /
          action::reject_invalid_batch = sml::state<errored>,

      sml::state<batching> / action::run_batch = sml::state<batch_decision>,
      sml::state<batch_decision>[guard::phase_failed{}] = sml::state<errored>,
      sml::state<batch_decision>[guard::phase_ok{}] / action::mark_done =
          sml::state<done>,

      sml::state<done> + sml::event<event::batch>[guard::valid_request{}] /
          action::begin_batch = sml::state<batching>,
      sml::state<done> + sml::event<event::batch>[guard::invalid_request{}] /
          action::reject_invalid_batch = sml::state<errored>,

      sml::state<errored> + sml::event<event::batch>[guard::valid_request{}] /
          action::begin_batch = sml::state<batching>,
      sml::state<errored> + sml::event<event::batch>[guard::invalid_request{}] /
          action::reject_invalid_batch = sml::state<errored>,

      sml::state<unexpected> + sml::event<event::batch>[guard::valid_request{}] /
          action::begin_batch = sml::state<batching>,
      sml::state<unexpected> + sml::event<event::batch>[guard::invalid_request{}] /
          action::reject_invalid_batch = sml::state<errored>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<batching> + sml::unexpected_event<sml::_> /
          action::on_unexpected = sml::state<unexpected>,
      sml::state<batch_decision> + sml::unexpected_event<sml::_> /
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

  bool process_event(const event::batch & ev) {
    namespace sml = boost::sml;
    const bool accepted = base_type::process_event(ev);
    if (this->is(sml::state<done>)) {
      action::dispatch_done(ev);
    } else if (this->is(sml::state<errored>) || this->is(sml::state<unexpected>)) {
      const int32_t err = context_.last_error == EMEL_OK ? EMEL_ERR_BACKEND : context_.last_error;
      action::dispatch_error(ev, err);
    }
    return accepted;
  }

  template <class event_type>
  bool process_event(const event_type & ev) {
    return base_type::process_event(ev);
  }

 private:
  action::context context_{};
};

}  // namespace emel::token::batcher
