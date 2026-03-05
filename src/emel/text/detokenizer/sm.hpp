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
struct binding_done_decision {};
struct binding_done_callback {};
struct binding_error_decision {};
struct binding_error_callback {};
struct idle {};
struct decoding {};
struct decode_token_validation {};
struct decode_piece_decision {};
struct decode_byte_capacity_decision {};
struct decode_byte_pending_decision {};
struct decode_byte_pending_write {};
struct decode_text_pending_decision {};
struct decode_text_pending_write {};
struct decode_text_write {};
struct decode_decision {};
struct detokenize_done_decision {};
struct detokenize_done_callback {};
struct detokenize_error_decision {};
struct detokenize_error_callback {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * text detokenizer orchestration model.
 *
 * state purposes:
 * - `uninitialized`: awaiting vocab bind.
 * - `binding`/`binding_decision`: validate and apply vocab binding.
 * - `binding_*_callback`: synchronous callback delivery before terminal state.
 * - `idle`: ready for detokenize requests.
 * - `decoding`/`decode_*`: explicit detokenize phases and branch decisions.
 * - `detokenize_*_callback`: synchronous callback delivery before terminal state.
 * - `done`/`errored`: terminal outcomes for the latest request.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_*` guards validate request payload pointers and bound state.
 * - explicit `*_error_*` guards branch on per-request typed error outputs.
 *
 * action side effects:
 * - `begin_detokenize` initializes request output fields.
 * - `append_byte_piece`/`write_pending_head_sequence`/`write_text_piece` execute decode kernels.
 * - `mark_done` finalizes success terminal status.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // External request validation.
        sml::state<binding> <= *sml::state<uninitialized> + sml::event<event::bind>
                   [ guard::valid_bind{} ] / action::begin_bind
      , sml::state<binding_error_decision> <= sml::state<uninitialized> + sml::event<event::bind>
                   [ guard::invalid_bind{} ] / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<uninitialized> + sml::event<event::detokenize>
                   / action::reject_detokenize

      , sml::state<binding> <= sml::state<idle> + sml::event<event::bind>
                   [ guard::valid_bind{} ] / action::begin_bind
      , sml::state<binding_error_decision> <= sml::state<idle> + sml::event<event::bind>
                   [ guard::invalid_bind{} ] / action::reject_bind
      , sml::state<decoding> <= sml::state<idle> + sml::event<event::detokenize>
                   [ guard::valid_detokenize{} ] / action::begin_detokenize
      , sml::state<detokenize_error_decision> <= sml::state<idle> + sml::event<event::detokenize>
                   [ guard::invalid_detokenize{} ] / action::reject_detokenize

      , sml::state<binding> <= sml::state<done> + sml::event<event::bind>
                   [ guard::valid_bind{} ] / action::begin_bind
      , sml::state<binding_error_decision> <= sml::state<done> + sml::event<event::bind>
                   [ guard::invalid_bind{} ] / action::reject_bind
      , sml::state<decoding> <= sml::state<done> + sml::event<event::detokenize>
                   [ guard::valid_detokenize{} ] / action::begin_detokenize
      , sml::state<detokenize_error_decision> <= sml::state<done> + sml::event<event::detokenize>
                   [ guard::invalid_detokenize{} ] / action::reject_detokenize

      , sml::state<binding> <= sml::state<errored> + sml::event<event::bind>
                   [ guard::valid_bind{} ] / action::begin_bind
      , sml::state<binding_error_decision> <= sml::state<errored> + sml::event<event::bind>
                   [ guard::invalid_bind{} ] / action::reject_bind
      , sml::state<decoding> <= sml::state<errored> + sml::event<event::detokenize>
                   [ guard::valid_detokenize{} ] / action::begin_detokenize
      , sml::state<detokenize_error_decision> <= sml::state<errored> + sml::event<event::detokenize>
                   [ guard::invalid_detokenize{} ] / action::reject_detokenize

      , sml::state<binding> <= sml::state<unexpected> + sml::event<event::bind>
                   [ guard::valid_bind{} ] / action::begin_bind
      , sml::state<binding_error_decision> <= sml::state<unexpected> + sml::event<event::bind>
                   [ guard::invalid_bind{} ] / action::reject_bind
      , sml::state<decoding> <= sml::state<unexpected> + sml::event<event::detokenize>
                   [ guard::valid_detokenize{} ] / action::begin_detokenize
      , sml::state<detokenize_error_decision> <= sml::state<unexpected> + sml::event<event::detokenize>
                   [ guard::invalid_detokenize{} ] / action::reject_detokenize

      //------------------------------------------------------------------------------//
      // Internal-state reentry rejection via unexpected external requests.
      , sml::state<binding_error_decision> <= sml::state<binding> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<binding> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<binding_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<binding_done_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<binding_done_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<binding_done_callback> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<binding_done_callback> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<binding_error_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<binding_error_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<binding_error_callback> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<binding_error_callback> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decoding> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decoding> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_token_validation> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_token_validation> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_piece_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_piece_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_byte_capacity_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_capacity_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_byte_pending_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_byte_pending_write> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_write> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_text_pending_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_text_pending_write> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_write> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_text_write> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_write> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<decode_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<detokenize_done_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<detokenize_done_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<detokenize_done_callback> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<detokenize_done_callback> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<detokenize_error_decision> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<detokenize_error_decision> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize
      , sml::state<binding_error_decision> <= sml::state<detokenize_error_callback> + sml::unexpected_event<event::bind>
                   / action::reject_bind
      , sml::state<detokenize_error_decision> <= sml::state<detokenize_error_callback> + sml::unexpected_event<event::detokenize>
                   / action::reject_detokenize

      //------------------------------------------------------------------------------//
      // Internal phase progression.
      , sml::state<binding_decision> <= sml::state<binding> + sml::completion<event::bind>
                   / action::commit_bind
      , sml::state<binding_done_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_none{} ]
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_invalid_request{} ]
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_model_invalid{} ]
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_backend_error{} ]
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_internal_error{} ]
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_untracked{} ]
      , sml::state<binding_error_decision> <= sml::state<binding_decision> + sml::completion<event::bind>
                   [ guard::bind_error_unknown{} ]
      , sml::state<binding_done_callback> <= sml::state<binding_done_decision> + sml::completion<event::bind>
                   [ guard::has_bind_done_callback{} ] / action::notify_bind_done
      , sml::state<idle> <= sml::state<binding_done_decision> + sml::completion<event::bind>
                   [ guard::no_bind_done_callback{} ]
      , sml::state<idle> <= sml::state<binding_done_callback> + sml::completion<event::bind>
      , sml::state<binding_error_callback> <= sml::state<binding_error_decision> + sml::completion<event::bind>
                   [ guard::has_bind_error_callback{} ] / action::notify_bind_error
      , sml::state<errored> <= sml::state<binding_error_decision> + sml::completion<event::bind>
                   [ guard::no_bind_error_callback{} ]
      , sml::state<errored> <= sml::state<binding_error_callback> + sml::completion<event::bind>

      , sml::state<decode_token_validation> <= sml::state<decoding> + sml::completion<event::detokenize>
      , sml::state<decode_piece_decision> <= sml::state<decode_token_validation> + sml::completion<event::detokenize>
                   [ guard::detokenize_token_in_vocab{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_token_validation> + sml::completion<event::detokenize>
                   [ guard::detokenize_token_out_of_vocab{} ] / action::mark_model_invalid
      , sml::state<detokenize_done_decision> <= sml::state<decode_piece_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_skip_special_piece{} ] / action::mark_done
      , sml::state<decode_byte_capacity_decision> <= sml::state<decode_piece_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_byte_piece{} ]
      , sml::state<decode_text_pending_decision> <= sml::state<decode_piece_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_text_piece{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_piece_decision> + sml::completion<event::detokenize>
                   / action::mark_internal_error
      , sml::state<decode_byte_pending_decision> <= sml::state<decode_byte_capacity_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_has_capacity_for_byte{} ] / action::append_byte_piece
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_capacity_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_no_capacity_for_byte{} ] / action::mark_invalid_pending_full
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_invalid_request{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_model_invalid{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_backend_error{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_internal_error{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_untracked{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_unknown{} ]
      , sml::state<decode_byte_pending_write> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_head_complete{} ] / action::write_pending_head_sequence
      , sml::state<decode_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_empty{} ]
      , sml::state<decode_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_head_incomplete{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_byte_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_head_invalid{} ] / action::mark_invalid_pending_sequence
      , sml::state<decode_byte_pending_decision> <= sml::state<decode_byte_pending_write> + sml::completion<event::detokenize>

      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_invalid_request{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_model_invalid{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_backend_error{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_internal_error{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_untracked{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_unknown{} ]
      , sml::state<decode_text_pending_write> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_head_complete{} ] / action::write_pending_head_sequence
      , sml::state<decode_text_write> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_empty{} ] / action::write_text_piece
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_head_incomplete{} ] / action::mark_invalid_pending_not_empty
      , sml::state<detokenize_error_decision> <= sml::state<decode_text_pending_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_pending_head_invalid{} ] / action::mark_invalid_pending_sequence
      , sml::state<decode_text_pending_decision> <= sml::state<decode_text_pending_write> + sml::completion<event::detokenize>
      , sml::state<decode_decision> <= sml::state<decode_text_write> + sml::completion<event::detokenize>
      , sml::state<detokenize_done_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_none{} ] / action::mark_done
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_invalid_request{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_model_invalid{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_backend_error{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_internal_error{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_untracked{} ]
      , sml::state<detokenize_error_decision> <= sml::state<decode_decision> + sml::completion<event::detokenize>
                   [ guard::detokenize_error_unknown{} ]
      , sml::state<detokenize_done_callback> <= sml::state<detokenize_done_decision> + sml::completion<event::detokenize>
                   [ guard::has_detokenize_done_callback{} ]
      , sml::state<done> <= sml::state<detokenize_done_decision> + sml::completion<event::detokenize>
                   [ guard::no_detokenize_done_callback{} ]
      , sml::state<done> <= sml::state<detokenize_done_callback> + sml::completion<event::detokenize>
                   / action::notify_detokenize_done
      , sml::state<detokenize_error_callback> <= sml::state<detokenize_error_decision> + sml::completion<event::detokenize>
                   [ guard::has_detokenize_error_callback{} ] / action::notify_detokenize_error
      , sml::state<errored> <= sml::state<detokenize_error_decision> + sml::completion<event::detokenize>
                   [ guard::no_detokenize_error_callback{} ]
      , sml::state<errored> <= sml::state<detokenize_error_callback> + sml::completion<event::detokenize>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<unexpected> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_done_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_done_callback> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_error_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_error_callback> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<idle> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decoding> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_token_validation> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_piece_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_byte_capacity_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_byte_pending_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_byte_pending_write> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_text_pending_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_text_pending_write> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_text_write> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<detokenize_done_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<detokenize_done_callback> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<detokenize_error_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<detokenize_error_callback> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() = default;

  bool process_event(const event::bind & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool at_idle = this->is(sml::state<idle>);
    return accepted & at_idle;
  }

  bool process_event(const event::detokenize & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool at_done = this->is(sml::state<done>);
    return accepted & at_done;
  }

  using base_type::process_event;
  using base_type::is;
  using base_type::visit_current_states;
};

}  // namespace emel::text::detokenizer
