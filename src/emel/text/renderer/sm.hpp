#pragma once

/*
design doc: docs/designs/text/renderer.design.md
 ---
 title: text/renderer architecture design
 status: draft
 ---
 
 # text/renderer architecture design
 
 this document defines the text/renderer actor. it acts as a domain-specific output consumer for the
 modality-agnostic `generator`. it receives raw token IDs, translates them to text, and handles
 string-based stopping criteria and streaming formats.
 
 ## role
 - act as an injected dependency (or external consumer) for the `generator`.
 - receive raw `token_id` streams from the generator's sampling phase.
 - own the `text/detokenizer` codec to translate tokens into utf-8 bytes.
 - handle text-domain complexities: utf-8 boundary buffering, whitespace stripping, and string-based
   stop sequence matching.
 
 ## architecture shift: omnimodal decoupling
 to support omnimodal generation (text, audio, vision), the core `generator` must remain completely
 ignorant of what a `token_id` represents. it cannot own text-specific formatting or stop sequences.
 
 the `generator` is passed a `renderer` (e.g., a `text/renderer` or an `audio/renderer`). when the
 generator samples a new token, it passes it to the renderer. if the renderer detects that the newly
 emitted text matches a user-defined string stop sequence (e.g., `"\nUser:"`), the renderer signals
 back to the generator to halt that sequence.
 
 ## events
 - `event::bind`
   - inputs: `vocab`, renderer options (e.g., stop sequences, strip whitespace flags), and optional callbacks.
   - outputs: invokes callback upon successfully binding state ready to render.
 - `event::render` (called by the generator post-sampling)
   - inputs: `token_id`, sequence ID, output buffers (for streaming to the user), and optional callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: populates caller-provided buffers with translated utf-8 bytes (if any) and a `sequence_status` flag,
     invoking the appropriate callback before returning to prevent context reading.
 - `event::flush`
   - inputs: sequence ID, output buffers, and optional callbacks.
   - outputs: forces the emission of any pending bytes into the output buffers and invokes the callback.
 
 ## state model
 
 ```text
 uninitialized ──► binding ──► idle
                                │
 idle ──► rendering ──► render_decision ──► (idle | errored)
   ▲                                           │
   └───────────────────────────────────────────┘
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `binding` — storing vocab references and compiling stop sequence patterns.
 - `idle` — waiting for a `token_id` from the generator.
 - `rendering` — passing the token to the `text/detokenizer` and buffering the result.
 - `render_decision` — evaluating the newly translated text against the stop sequence list.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 1. **utf-8 stream management:** modern models use byte-fallback tokens. a single `token_id` might be
    a partial utf-8 character (e.g., `0xE2`). the renderer maintains a tiny pending byte buffer per
    sequence and only flushes valid, complete utf-8 strings to the user's output buffer.
 2. **stop sequence matching:** evaluate the rolling text window against user-provided stop strings.
    if a match occurs, truncate the output and return a `stop_sequence_matched` status to the generator.
 3. **formatting:** handle optional policies like stripping leading spaces from the very first token
    generated.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_CAPACITY` — output buffer too small for the rendered bytes.
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid sequence id.
 - `EMEL_ERR_STOPPED` — render attempted on a sequence that has already been stopped.
*/


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
