# text/detokenizer architecture design (draft)

this document defines a detokenizer actor that converts token ids back into text for renderer
outputs. it is intentionally separate from the tokenizer encoder to keep the codec split and avoid
orthogonal regions in a single SM.

## role
- text/detokenizer is a codec-stage actor: token ids -> text/bytes.
- detokenizer is model/vocab-aware and must honor special-token and byte-fallback rules.

## public interface (proposed)
- `event::bind` (vocab binding)
  - inputs: `vocab`, `error_out`, optional callbacks.
  - outputs: bound state ready for decode.
- `event::detokenize` (decode tokens)
  - inputs: `vocab`, `token_ids`, `token_count`, `emit_special`, `utf8_out`, `utf8_capacity`,
    `utf8_count_out`, `error_out`, optional callbacks.
  - outputs: written utf8 bytes + count; errors via `error_out`.
- streaming support (optional):
  - `event::flush` to emit any pending partial UTF-8 bytes.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `decoding` -> `decode_decision` -> `done`.
- `errored` and `unexpected` terminal error states with recovery on next valid bind.

## decoding rules
- token -> piece mapping uses vocab metadata (token text, flags, byte-fallback).
- optional `emit_special` controls whether special tokens are emitted as text or skipped.
- byte-fallback tokens are accumulated into a byte buffer and only emitted when forming valid
  UTF-8 sequences.
- invalid token ids or insufficient output capacity are errors.

## context (draft)
- bound vocab pointer and model kind.
- small pending-byte buffer for UTF-8 boundary handling (fixed size, no heap).
- counters: output count, last error.

## invariants
- no allocations during dispatch; all buffers fixed-size.
- no self-dispatch; bounded RTC work per request.
- output is written only to caller-provided buffers.

## open questions
- how to represent special-token rendering (literal string vs placeholder vs skip)?
- maximum pending-byte buffer size needed for UTF-8 assembly.
