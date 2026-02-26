#pragma once

/*
design doc: docs/designs/token/batcher.design.md
 ---
 title: token/batcher architecture design
 status: draft
 ---
 
 # token/batcher architecture design
 
 this document defines token/batcher. it acts as the strict gateway between raw user input and the
 generator's execution pipeline. it sanitizes, auto-populates, and validates token arrays, emitting a
 canonical `token::batch` that the stateless `batch/planner` can trust completely.
 
 ## role
 - act as a strict firewall: reject malformed token inputs (out-of-bounds IDs, invalid sequence
   positions) before they reach the graph or memory systems.
 - auto-populate missing dimensions (e.g., sequence IDs, position IDs, logit masks) using "do what I
   mean" defaults based on the current `memory::any` state.
 - emit a `token::batch`—a canonical, validated view of token IDs and metadata that serves as the
   single source of truth for the current generator cycle.
 
 ## architecture shift: the strict gateway
 in dynamic systems (like `llama.cpp`), token validation is often intertwined with memory allocation
 and planning (e.g., checking vocabulary bounds while simultaneously copying arrays into a new struct).
 
 in `emel`, because the `batch/planner` is purely stateless and mathematical, the `token/batcher` must
 guarantee that the `token::batch` it produces is logically sound. if the batcher succeeds, the
 planner is guaranteed to be able to slice it without encountering semantic errors (like a sequence
 jumping backwards in time).
 
 ## events
 - `event::batch`
   - inputs: raw token IDs + token count, optional sequence IDs, optional position IDs,
     optional logit targets (output masks), and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: sanitizes the input, populates the canonical `token::batch` structures in the caller-provided
     buffers, and invokes the appropriate callback before returning, completely avoiding context reads.
 
 ## state model
 
 ```text
 uninitialized ──► idle
                     │
 idle ──► populating ──► validating ──► (done | errored)
   ▲                                       │
   └───────────────────────────────────────┘
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `idle` — waiting for a raw token array.
 - `populating` — filling in missing metadata (e.g., if positions are omitted, query `memory::any`
   for the current sequence length and auto-increment).
 - `validating` — strictly enforcing logical continuity and vocabulary bounds.
 - `done` — validation passed, transitions back to `idle` emitting `events::batch_done`.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 
 1. **auto-population (the "do what I mean" phase):**
    - **sequence IDs:** if omitted, default all tokens to a primary sequence (e.g., sequence `0`).
    - **positions:** if `position_ids` are omitted, query the `memory::any` interface to find the
      current length of the requested sequence(s), and auto-increment from there.
    - **logit masks:** if the user doesn't specify which tokens need output logits, default to `false`
      for all tokens *except* the very last token in the sequence (saving massive kernel compute).
 
 2. **strict validation (the firewall):**
    - **vocabulary bounds:** reject any `token_id` that is `< 0` or `>= vocab.size()`.
    - **sequence continuity:** ensure the `position_ids` for any given sequence are strictly
      monotonically increasing. (e.g., reject `pos: [4, 5, 2, 3]` for sequence A).
    - **coupling constraints:** if two sequences share a token (e.g., during prompt sharing or
      prefix caching), ensure they agree on the position of that token. if they have diverged, reject.
 
 3. **canonical emission:**
    - the resulting `token::batch` is not a copy of the data, but a structured, validated view
      (often just pointers to the user's arrays, plus the auto-populated metadata arrays).
    - this `token::batch` is handed directly to the `batch/planner` for zero-copy slicing.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_CAPACITY` — output buffer too small for the batched token data.
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid or duplicate sequence ids, missing position seed, or out-of-bounds token ids.
 - `EMEL_ERR_INTERNAL` — internal invariant violation.
*/


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
