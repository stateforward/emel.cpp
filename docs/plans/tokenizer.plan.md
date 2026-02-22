# tokenizer plan (rolling)

this document captures the agreed tokenizer architecture and outstanding work. it is a planning
artifact only; `src/` machines remain the source of truth and any structural changes require
explicit approval.

## scope
- tokenize text into token ids with special-token handling and model-aware encoding.
- coordinate preprocessing (fragment partitioning) and encoder selection.
- provide a stable C ABI surface via `emel/emel.h`.

## decisions
- tokenizer remains the orchestration owner; preprocessing and encoding stay as composed stages.
- tokenizer owns preprocessor selection; no separate selector component.
- preprocessing is modeled as a component with variant SMs under `tokenizer/preprocessor/*`.
- encoding is modeled as encoder SMs under `encoder/*` with a stable `encoder::event::encode` API.
- tokenizer is bound to a loaded model/vocab; encoder binding happens at init (no per-request select).
- binding is performed via an explicit tokenizer `bind` event before tokenization.
- tokenizer outputs are written only through the request payload (no persistent output buffers).
- tokenizer errors are surfaced via `_error` events and error_out fields only.

## architecture (top-level flow)
1. validate request and reset context.
2. build special-token cache for the vocab.
3. preprocess text into fragments (raw spans + special tokens).
4. apply optional prefix (BOS) with capacity validation.
5. encode fragments in a bounded RTC loop.
6. apply optional suffix (SEP/EOS) with capacity validation.
7. finalize and dispatch done/error.

## components
- `tokenizer/sm.hpp`: primary orchestration for tokenization requests.
- `tokenizer/actions.hpp`: request capture, fragment bookkeeping, prefix/suffix append, encoder dispatch.
- `tokenizer/guards.hpp`: validation and capacity checks.
- `tokenizer/preprocessor/*`: variant preprocessors that emit fragment lists.
- `encoder/*`: model-specific encoders (SPM/BPE/WPM/UGM/RWKV/PLAMO2/fallback).
- `tokenizer/context.hpp`: owns runtime data (fragments, encoder contexts, counters).

## tokenizer state model (draft)
- init flow:
  `uninitialized` -> `binding_preprocessor` -> `binding_encoder` -> `idle`
- `idle`: ready for a tokenize request; validates request and captures inputs.
- `preprocessing`: dispatch preprocessor for fragment generation.
- `preprocess_decision`: branch on preprocessor success/failure.
- `prefix_decision`: apply optional BOS or reject on capacity/invalid id.
- `encoding_ready`: decide next fragment or move to suffix.
- `encoding_token_fragment`: append a token fragment.
- `encoding_raw_fragment`: encode a raw-text fragment through encoder.
- `encoding_decision`: branch on encode success/failure, loop back if ok.
- `suffix_decision`: apply optional SEP/EOS or reject on capacity/invalid id.
- `finalizing`: mark success and prepare outputs.
- `done`: terminal success for the request.
- `errored`: terminal failure for the request.
- `unexpected`: sequencing contract violation.

## preprocessing model
- preprocessors are responsible for fragment generation only.
- special-token matching is done before encoding and can honor lstrip/rstrip flags.
- BPE pre-splitting is handled by the BPE preprocessor variant using regex keyed by
  `tokenizer_pre_id`.
- non-BPE preprocessors may be identity or minimal normalization, but must emit fragments with
  consistent semantics.
- preprocessor output is a bounded fragment list (no allocations during dispatch).

## encoding model
- encoder binding is done at init from `vocab->tokenizer_model_id` with a fallback encoder.
- encoders must treat input fragments as authoritative and never re-run special token parsing.
- encoders accept pre-tokenized BPE fragments via `event::encode.pretokenized`.
- fragment encoding is a bounded RTC loop with explicit capacity checks.
- prefix/suffix handling is centralized in the tokenizer (not encoder-specific code paths).

## data contracts
- inputs: `event::tokenize` carries vocab pointer, text view, flags, output buffers, and callbacks.
- inputs: `event::bind` carries the vocab pointer and error output for binding.
- preprocessing output: `fragment` array with `raw_text` spans or `token` ids.
- outputs: `token_ids_out` and `token_count_out` only; no context-owned output buffers.
- errors: `error_out` and `_error` events; context holds only the latest error code.

## invariants
- no allocations during dispatch; any required allocation must happen at init.
- no actor re-entrancy; no self `process_event` calls in guards/actions.
- all internal progress uses anonymous transitions and is statically bounded.
- unexpected events are always surfaced via `sml::unexpected_event` handling.

## testing strategy
- unit tests per preprocessor variant and per encoder.
- tokenizer orchestration tests validate prefix/suffix, capacity errors, and fragment sequencing.
- SML introspection used for state assertions.
- coverage target >= 90% via `scripts/quality_gates.sh`.

## open questions
- do we expose a standalone `tokenizer::preprocessor::sm` alias in `src/emel/machines.hpp`?
- where should byte-fallback behavior live (`tokenizer/byte_fallback` vs encoder detail)?
- how do we want to surface tokenizer preprocessor policy selection for callers?
