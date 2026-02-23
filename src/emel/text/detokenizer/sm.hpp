#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/text/detokenizer/actions.hpp"
#include "emel/text/detokenizer/events.hpp"
#include "emel/text/detokenizer/guards.hpp"

namespace emel::text::detokenizer {

struct uninitialized {};
struct binding {};
struct binding_decision {};
struct idle {};
struct decoding {};
struct decode_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * text detokenizer orchestration model.
 *
 * state purposes:
 * - `uninitialized`: awaiting vocab bind.
 * - `binding`/`binding_decision`: validate and apply vocab binding.
 * - `idle`: ready for detokenize requests.
 * - `decoding`/`decode_decision`: translate token id into output bytes.
 * - `done`/`errored`: terminal outcomes for the latest request.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_*` guards validate request payload pointers and bound state.
 * - `phase_*` guards branch on action-set error codes.
 *
 * action side effects:
 * - `begin_detokenize` captures output/pending buffers from request.
 * - `decode_token` emits bytes and updates pending utf-8 fragments.
 * - `mark_done`/`ensure_last_error` finalize terminal status.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<uninitialized> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
            sml::state<binding>,
        sml::state<uninitialized> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<errored>,
        sml::state<uninitialized> + sml::event<event::detokenize> /
            action::reject_detokenize = sml::state<errored>,

        sml::state<idle> + sml::event<event::bind>[guard::valid_bind{}] /
                               action::begin_bind = sml::state<binding>,
        sml::state<idle> + sml::event<event::bind>[guard::invalid_bind{}] /
                               action::reject_bind = sml::state<errored>,
        sml::state<idle> +
            sml::event<event::detokenize>[guard::valid_detokenize{}] /
                action::begin_detokenize = sml::state<decoding>,
        sml::state<idle> +
            sml::event<event::detokenize>[guard::invalid_detokenize{}] /
                action::reject_detokenize = sml::state<errored>,

        sml::state<done> + sml::event<event::bind>[guard::valid_bind{}] /
                               action::begin_bind = sml::state<binding>,
        sml::state<done> + sml::event<event::bind>[guard::invalid_bind{}] /
                               action::reject_bind = sml::state<errored>,
        sml::state<done> +
            sml::event<event::detokenize>[guard::valid_detokenize{}] /
                action::begin_detokenize = sml::state<decoding>,
        sml::state<done> +
            sml::event<event::detokenize>[guard::invalid_detokenize{}] /
                action::reject_detokenize = sml::state<errored>,

        sml::state<errored> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
                sml::state<binding>,
        sml::state<errored> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<errored>,
        sml::state<errored> +
            sml::event<event::detokenize>[guard::valid_detokenize{}] /
                action::begin_detokenize = sml::state<decoding>,
        sml::state<errored> +
            sml::event<event::detokenize>[guard::invalid_detokenize{}] /
                action::reject_detokenize = sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
                sml::state<binding>,
        sml::state<unexpected> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<event::detokenize>[guard::valid_detokenize{}] /
                action::begin_detokenize = sml::state<decoding>,
        sml::state<unexpected> +
            sml::event<event::detokenize>[guard::invalid_detokenize{}] /
                action::reject_detokenize = sml::state<unexpected>,

        sml::state<binding> / action::commit_bind = sml::state<binding_decision>,
        sml::state<binding_decision>[guard::phase_ok{}] = sml::state<idle>,
        sml::state<binding_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<decoding> / action::decode_token = sml::state<decode_decision>,
        sml::state<decode_decision>[guard::phase_ok{}] / action::mark_done =
            sml::state<done>,
        sml::state<decode_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<uninitialized> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<binding> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<binding_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<idle> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<decoding> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<decode_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<unexpected> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::bind & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<idle>);
    const int32_t err = ok ? EMEL_OK
                           : (context_.last_error != EMEL_OK ? context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::binding_done{&ev});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::binding_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

  bool process_event(const event::detokenize & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = ok ? EMEL_OK
                           : (context_.last_error != EMEL_OK ? context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = context_.output_length;
    }
    if (ev.pending_length_out != nullptr) {
      *ev.pending_length_out = context_.pending_length;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(
            ev.owner_sm,
            events::detokenize_done{&ev, context_.output_length, context_.pending_length});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::detokenize_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  action::context context_{};
};

}  // namespace emel::text::detokenizer
