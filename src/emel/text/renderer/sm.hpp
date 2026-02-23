#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/text/renderer/actions.hpp"
#include "emel/text/renderer/events.hpp"
#include "emel/text/renderer/guards.hpp"

namespace emel::text::renderer {

struct uninitialized {};
struct binding {};
struct binding_decision {};
struct idle {};
struct rendering {};
struct render_decision {};
struct flushing {};
struct flush_decision {};
struct done {};
struct errored {};
struct unexpected {};

/*
renderer architecture notes (single source of truth)

state purpose
- uninitialized: awaiting dependency and vocab binding.
- binding/binding_decision: binds detokenizer dependency for the selected vocab.
- idle: ready for render and flush requests.
- rendering/render_decision: detokenizes one token and applies stop matching.
- flushing/flush_decision: emits buffered bytes (utf-8 pending + stop holdback).
- done/errored: terminal outcomes for the latest request.
- unexpected: sequencing contract violation.

key invariants
- per-sequence utf-8 pending bytes and stop holdback are stored in renderer context.
- detokenizer stays stateless and receives all pending state via event payloads.
- output bytes are written only to caller-provided buffers.

guard semantics
- valid_bind: dependency pointers and bind inputs are present.
- valid_render/valid_flush: output pointers and sequence id are valid.
- phase_ok/phase_failed: branch on action-set error codes.

action side effects
- begin_bind/bind_detokenizer: configure dependencies and bind child actor.
- begin_render/run_render: decode token and apply strip/stop policy.
- begin_flush/run_flush: force buffered output emission.
- mark_done/ensure_last_error: finalize terminal request result.
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
        sml::state<uninitialized> + sml::event<event::render> /
            action::reject_render = sml::state<errored>,
        sml::state<uninitialized> + sml::event<event::flush> /
            action::reject_flush = sml::state<errored>,

        sml::state<idle> + sml::event<event::bind>[guard::valid_bind{}] /
                               action::begin_bind = sml::state<binding>,
        sml::state<idle> + sml::event<event::bind>[guard::invalid_bind{}] /
                               action::reject_bind = sml::state<errored>,
        sml::state<idle> + sml::event<event::render>[guard::valid_render{}] /
                               action::begin_render = sml::state<rendering>,
        sml::state<idle> +
            sml::event<event::render>[guard::invalid_render{}] /
                action::reject_render = sml::state<errored>,
        sml::state<idle> + sml::event<event::flush>[guard::valid_flush{}] /
                               action::begin_flush = sml::state<flushing>,
        sml::state<idle> + sml::event<event::flush>[guard::invalid_flush{}] /
                               action::reject_flush = sml::state<errored>,

        sml::state<done> + sml::event<event::bind>[guard::valid_bind{}] /
                               action::begin_bind = sml::state<binding>,
        sml::state<done> + sml::event<event::bind>[guard::invalid_bind{}] /
                               action::reject_bind = sml::state<errored>,
        sml::state<done> + sml::event<event::render>[guard::valid_render{}] /
                               action::begin_render = sml::state<rendering>,
        sml::state<done> +
            sml::event<event::render>[guard::invalid_render{}] /
                action::reject_render = sml::state<errored>,
        sml::state<done> + sml::event<event::flush>[guard::valid_flush{}] /
                               action::begin_flush = sml::state<flushing>,
        sml::state<done> + sml::event<event::flush>[guard::invalid_flush{}] /
                               action::reject_flush = sml::state<errored>,

        sml::state<errored> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
                sml::state<binding>,
        sml::state<errored> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<errored>,
        sml::state<errored> +
            sml::event<event::render>[guard::valid_render{}] /
                action::begin_render = sml::state<rendering>,
        sml::state<errored> +
            sml::event<event::render>[guard::invalid_render{}] /
                action::reject_render = sml::state<errored>,
        sml::state<errored> +
            sml::event<event::flush>[guard::valid_flush{}] /
                action::begin_flush = sml::state<flushing>,
        sml::state<errored> +
            sml::event<event::flush>[guard::invalid_flush{}] /
                action::reject_flush = sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::bind>[guard::valid_bind{}] / action::begin_bind =
                sml::state<binding>,
        sml::state<unexpected> +
            sml::event<event::bind>[guard::invalid_bind{}] /
                action::reject_bind = sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<event::render>[guard::valid_render{}] /
                action::begin_render = sml::state<rendering>,
        sml::state<unexpected> +
            sml::event<event::render>[guard::invalid_render{}] /
                action::reject_render = sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<event::flush>[guard::valid_flush{}] /
                action::begin_flush = sml::state<flushing>,
        sml::state<unexpected> +
            sml::event<event::flush>[guard::invalid_flush{}] /
                action::reject_flush = sml::state<unexpected>,

        sml::state<binding> / action::bind_detokenizer =
            sml::state<binding_decision>,
        sml::state<binding_decision>[guard::phase_ok{}] = sml::state<idle>,
        sml::state<binding_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<rendering> / action::run_render =
            sml::state<render_decision>,
        sml::state<render_decision>[guard::phase_ok{}] / action::mark_done =
            sml::state<done>,
        sml::state<render_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<flushing> / action::run_flush = sml::state<flush_decision>,
        sml::state<flush_decision>[guard::phase_ok{}] / action::mark_done =
            sml::state<done>,
        sml::state<flush_decision>[guard::phase_failed{}] /
            action::ensure_last_error = sml::state<errored>,

        sml::state<uninitialized> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<binding> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<binding_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<idle> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<rendering> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<render_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<flushing> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<flush_decision> + sml::unexpected_event<sml::_> /
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

  bool process_event(const event::render & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = ok ? EMEL_OK
                           : (context_.last_error != EMEL_OK ? context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = context_.output_length;
    }
    if (ev.status_out != nullptr) {
      *ev.status_out = context_.status;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(
            ev.owner_sm,
            events::rendering_done{&ev, context_.output_length, context_.status});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::rendering_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

  bool process_event(const event::flush & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = ok ? EMEL_OK
                           : (context_.last_error != EMEL_OK ? context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = context_.output_length;
    }
    if (ev.status_out != nullptr) {
      *ev.status_out = context_.status;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(
            ev.owner_sm,
            events::flush_done{&ev, context_.output_length, context_.status});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::flush_error{&ev, err});
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

}  // namespace emel::text::renderer
