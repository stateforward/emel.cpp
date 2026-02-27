#pragma once

/*
design doc: docs/designs/text/conditioner.design.md
 ---
 title: text/conditioner architecture design
 status: draft
 ---
 
 # text/conditioner architecture design
 
 this document defines the text/conditioner actor family. conditioners act as domain-specific input
 processors for the modality-agnostic `generator`, formatting user intents into raw text strings and
 orchestrating the `text/tokenizer` to produce raw token arrays.
 
 ## role
 - act as an injected dependency for the `generator`.
 - provide a configurable pipeline to format user inputs (raw text or structured chat messages) into
   model-specific prompt strings.
 - own and orchestrate the `text/tokenizer` codec to translate those text strings into `token_id`s.
 - yield raw, unsanitized token arrays back to the `generator`.
 
 ## architecture shift: dependency injection (`text/formatter`)
 to provide both power-user flexibility and high-level ergonomics without complicating the SML state
 machine, the `text/conditioner` relies on **explicit dependency injection**.
 
 it does not format text itself. instead, it owns two injected dependencies:
 1. **`text/formatter`:** a pure, stateless class that converts user input (e.g., raw strings or
    chat message arrays) into a single, contiguous string.
 2. **`text/tokenizer`:** an SML actor that converts that string into `token_id`s.
 
 when the user swaps a `formatter::raw` for a `formatter::chat`, the `conditioner`'s SML logic
 remains completely unchanged. it simply orchestrates the handoff: Input -> Formatter -> Tokenizer -> Yield.
 
 crucially, the conditioner does **not** perform batch slicing, sequence mapping, or memory validation.
 it yields a raw array of `token_id`s to the `generator`, which uses its internal `token/batcher` to
 handle the mathematical sanitation.
 
 ## events
 - `event::bind`
   - inputs: `vocab`, injected `formatter` instance, and optional synchronous callbacks.
   - outputs: invokes callback upon successfully binding state ready to condition inputs.
 - `event::prepare` (called by the user or the generator's step loop)
   - inputs: user prompt (domain-specific input payload) and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: passes the payload to the formatter, dispatches the resulting string to the tokenizer,
     populates the caller-provided buffer with a raw array of `token_id`s, and invokes the callback before returning.
 
 ## state model (general)
 
 ```text
 uninitialized ──► binding ──► idle
                                │
 idle ──► formatting ──► tokenizing ──► (idle | errored)
   ▲                                           │
   └───────────────────────────────────────────┘
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `binding` — storing vocab references and binding the injected formatter.
 - `idle` — waiting for a user prompt.
 - `formatting` — synchronously calling the injected `text/formatter` to generate the prompt string.
 - `tokenizing` — dispatching `event::tokenize` to the owned `text/tokenizer` to convert the string
   into `token_id`s.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 1. **flexible formatting:** provide a clean interface for users to choose their level of abstraction
    (raw text vs. auto-templated chat arrays) without muddying the core engine.
 2. **tokenizer orchestration:** manage the synchronous invocation of the `text/tokenizer` actor,
    handling special-token parsing flags based on the formatting stage.
 3. **raw yield:** provide a flat array of `token_id`s to the `generator`, delegating all sequence
    management and memory bounding to the generator's internal `token/batcher`.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_CAPACITY` — output buffer too small for the formatted prompt or token array.
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid template handle or malformed input.
 - any error code propagated from the owned `text/tokenizer` is forwarded unchanged.
*/


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

    action::clear_prepare_request(this->context_);
    return accepted && ok;
  }

  bool process_event(const event::prepare & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = ok ? EMEL_OK
                           : (this->context_.last_error != EMEL_OK ? this->context_.last_error
                                                             : EMEL_ERR_BACKEND);

    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = this->context_.token_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm,
                         events::conditioning_done{&ev, this->context_.token_count});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::conditioning_error{&ev, err});
      }
    }

    action::clear_prepare_request(this->context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return this->context_.last_error; }

 private:
};

}  // namespace emel::text::conditioner
