# text/conditioner architecture design (draft)

this document defines the text/conditioner actor. it owns prompt conditioning and tokenization and
prepares inputs for generator.

## role
- text/conditioner is the input-side orchestrator for text generation.
- it owns the tokenizer codec and prepares conditioned batch inputs (tokens, masks, positions).
- it may merge optional conditioning context from other modalities (provided by a higher-level
  orchestrator).

## ownership
- owns: `text/tokenizer`.
- consumes: optional external conditioning context (e.g., cross-attn memory handles).
- produces: conditioned token batch inputs suitable for generator.

## public interface (draft)
- `event::prepare`:
  - inputs: text prompt(s), tokenizer options, output buffers for tokens/metadata, optional
    conditioning context handle.
  - outputs: token ids, per-token metadata (positions, seq ids, masks), error/status.

## state model (draft)
- `uninitialized` -> `binding_tokenizer` -> `idle`.
- `idle` -> `conditioning` -> `condition_decision` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- bind tokenizer based on model metadata.
- run tokenization and populate batch fields required by downstream consumers.
- validate output capacity and sequence continuity.

## conditioned batch contract (draft)
- required: token ids + token count.
- optional: positions array (otherwise generated downstream).
- optional: seq masks + primary ids for multi-sequence batches.
- output selection fields (output mask, output_all, enforce_single_output_per_seq).

## open questions
- should conditioner own batch sanitization/splitting, or only prepare raw token batch inputs?
- how conditioning context is represented (opaque handle vs structured buffers).
