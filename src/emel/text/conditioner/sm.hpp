#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/text/conditioner/actions.hpp"
#include "emel/text/conditioner/events.hpp"
#include "emel/text/conditioner/guards.hpp"

namespace emel::text::conditioner {

struct uninitialized {};
struct binding {};
struct binding_decision {};
struct idle {};
struct formatting {};
struct format_decision {};
struct tokenizing {};
struct tokenize_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * text conditioner orchestration model.
 *
 * state purposes:
 * - `uninitialized`: awaiting dependency and vocab binding.
 * - `binding`/`binding_decision`: bind tokenizer dependency for the selected vocab.
 * - `idle`: ready to prepare inputs.
 * - `formatting`/`format_decision`: run injected formatter into bounded buffer.
 * - `tokenizing`/`tokenize_decision`: dispatch formatted text to tokenizer.
 * - `done`/`errored`: terminal outcomes for the latest prepare request.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_bind`/`valid_prepare` validate API contracts and bound dependencies.
 * - `phase_*` guards branch on action-set error codes.
 *
 * action side effects:
 * - `begin_bind`/`bind_tokenizer` store dependencies and bind tokenizer.
 * - `begin_prepare` captures request outputs and flags.
 * - `run_format` formats input through injected formatter.
 * - `run_tokenize` emits raw token arrays via tokenizer.
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
        sml::state<uninitialized> + sml::event<event::prepare> /
            action::reject_prepare = sml::state<errored>,

        sml::state<idle> + sml::event<event::bind>[guard::valid_bind{}] /
                               action::begin_bind = sml::state<binding>,
        sml::state<idle> + sml::event<event::bind>[guard::invalid_bind{}] /
                               action::reject_bind = sml::state<errored>,
        sml::state<idle> +
            sml::event<event::prepare>[guard::valid_prepare{}] /
                action::begin_prepare = sml::state<formatting>,
        sml::state<idle> +
            sml::event<event::prepare>[guard::invalid_prepare{}] /
                action::reject_prepare = sml::state<errored>,

        sml::state<done> + sml::event<event::bind>[guard::valid_bind{}] /
                               action::begin_bind = sml::state<binding>,
        sml::state<done> + sml::event<event::bind>[guard::invalid_bind{}] /
                               action::reject_bind = sml::state<errored>,
        sml::state<done> +
            sml::event<event::prepare>[guard::valid_prepare{}] /
                action::begin_prepare = sml::state<formatting>,
        sml::state<done> +
            sml::event<event::prepare>[guard::invalid_prepare{}] /
                action::reject_prepare = sml::state<errored>,

        sml::state<errored> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
                sml::state<binding>,
        sml::state<errored> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<errored>,
        sml::state<errored> +
            sml::event<event::prepare>[guard::valid_prepare{}] /
                action::begin_prepare = sml::state<formatting>,
        sml::state<errored> +
            sml::event<event::prepare>[guard::invalid_prepare{}] /
                action::reject_prepare = sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
                sml::state<binding>,
        sml::state<unexpected> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<event::prepare>[guard::valid_prepare{}] /
                action::begin_prepare = sml::state<formatting>,
        sml::state<unexpected> +
            sml::event<event::prepare>[guard::invalid_prepare{}] /
                action::reject_prepare = sml::state<unexpected>,

        sml::state<binding> / action::bind_tokenizer =
            sml::state<binding_decision>,
        sml::state<binding_decision>[guard::phase_ok{}] = sml::state<idle>,
        sml::state<binding_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<formatting> / action::run_format =
            sml::state<format_decision>,
        sml::state<format_decision>[guard::phase_ok{}] =
            sml::state<tokenizing>,
        sml::state<format_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<tokenizing> / action::run_tokenize =
            sml::state<tokenize_decision>,
        sml::state<tokenize_decision>[guard::phase_ok{}] / action::mark_done =
            sml::state<done>,
        sml::state<tokenize_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<uninitialized> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<binding> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<binding_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<idle> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<formatting> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<format_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<tokenizing> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<tokenize_decision> + sml::unexpected_event<sml::_> /
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

    action::clear_prepare_request(context_);
    return accepted && ok;
  }

  bool process_event(const event::prepare & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = ok ? EMEL_OK
                           : (context_.last_error != EMEL_OK ? context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = context_.token_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm,
                         events::conditioning_done{&ev, context_.token_count});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::conditioning_error{&ev, err});
      }
    }

    action::clear_prepare_request(context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  action::context context_{};
};

}  // namespace emel::text::conditioner
