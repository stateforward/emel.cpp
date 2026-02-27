#pragma once

/*
design doc: docs/designs/text/detokenizer.design.md
 ---
 title: text/detokenizer architecture design
 status: draft
 ---
 
 # text/detokenizer architecture design
 
 this document defines the text/detokenizer actor. it is a pure, low-level codec stage that converts
 token IDs directly into raw bytes or text pieces based on the model's vocabulary.
 
 ## role
 - act as the pure translation layer: `token_id` -> `bytes`.
 - remain completely stateless across sequences and unaware of generation logic, streaming, or stop
   conditions.
 - honor model-specific vocabulary rules, including special-token handling and byte-fallback encoding.
 
 ## architecture shift: pure stateless codec
 in dynamic systems, a detokenizer often holds internal state (like a pending byte buffer for partial
 utf-8 characters) that ties it to a specific sequence.
 
 in `emel`, to support efficient batch rendering, the `text/detokenizer` actor itself is stateless.
 any required state (like a 4-byte pending utf-8 buffer) is owned by the parent `text/renderer` and
 passed into the detokenizer via the `event::detokenize` payload. this allows a single, pooled
 detokenizer actor to translate tokens for thousands of different sequences interchangeably.
 
 ## events
 - `event::bind`
   - inputs: `vocab` reference and optional synchronous callbacks.
   - outputs: invokes callback upon successfully binding state ready for decode.
 - `event::detokenize`
   - inputs: `token_id`, a reference to the renderer's pending byte buffer, `emit_special` flag,
     the output buffer capacity, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: writes utf-8 bytes to the output buffer, updates the pending byte buffer state, and
     invokes the appropriate callback before returning, completely avoiding context-reading race conditions.
 
 ## state model
 
 ```text
 uninitialized ──► binding ──► idle
                                │
 idle ──► decoding ──► decode_decision ──► (idle | errored)
   ▲                                          │
   └──────────────────────────────────────────┘
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `binding` — validating the vocab reference.
 - `idle` — waiting for a translation request.
 - `decoding` — looking up the `token_id` in the vocabulary and applying byte-fallback rules.
 - `decode_decision` — determining if the resulting bytes form a complete utf-8 sequence (combined
   with the pending buffer). if yes, write to output; if no, store in the pending buffer.
 - unexpected events route to `unexpected`.
 
 ## decoding rules
 1. **token-to-piece mapping:** uses vocab metadata to look up the text string for a given ID.
 2. **special tokens:** if `emit_special` is false, tokens flagged as `special` (like `<|eot_id|>`)
    result in zero output bytes.
 3. **byte-fallback accumulation:** if the token represents a raw byte (e.g., `<0xE2>`), it is
    appended to the renderer-provided pending buffer. the detokenizer only flushes bytes to the
    output buffer when they form a valid, complete utf-8 sequence.
 
 ## invariants
 - no allocations during dispatch.
 - output is written only to caller-provided buffers.
 - the actor holds no sequence-specific state between dispatches.
*/


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

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() : base_type() {}

  bool process_event(const event::bind & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<idle>);
    const int32_t err = ok ? EMEL_OK
                           : (this->context_.last_error != EMEL_OK ? this->context_.last_error
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

    action::clear_request(this->context_);
    return accepted && ok;
  }

  bool process_event(const event::detokenize & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = ok ? EMEL_OK
                           : (this->context_.last_error != EMEL_OK ? this->context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = this->context_.output_length;
    }
    if (ev.pending_length_out != nullptr) {
      *ev.pending_length_out = this->context_.pending_length;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(
            ev.owner_sm,
            events::detokenize_done{&ev, this->context_.output_length, this->context_.pending_length});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::detokenize_error{&ev, err});
      }
    }

    action::clear_request(this->context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return this->context_.last_error; }

 private:
};

}  // namespace emel::text::detokenizer
