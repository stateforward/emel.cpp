# text/renderer architecture design (draft)

this document defines the text/renderer actor. it owns detokenization.

## role
- text/renderer is the output-side orchestrator for text generation.
- it owns the detokenizer codec.
- it converts token streams into utf-8 text for streaming and final output.

## ownership
- owns: `text/detokenizer`.
- consumes: token ids + per-token metadata provided by generator.
- produces: rendered utf-8 text buffers and completion status.

## public interface (draft)
- `event::bind`:
  - inputs: model metadata, vocab, renderer options, `error_out`.
  - outputs: bound state ready to render.
- `event::render`:
  - inputs: token ids, token count, optional row/seq mapping, emit policy, output buffers,
    `error_out`.
  - outputs: utf-8 bytes + count; completion status.
- `event::flush` (optional):
  - inputs: output buffers, `error_out`.
  - outputs: pending utf-8 bytes.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `rendering` -> `render_decision` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- apply special-token handling and skip rules.
- coordinate detokenizer output policies.
- manage streaming boundaries without allocations.

## open questions
- where should stop-sequence handling live (renderer vs generator)?
