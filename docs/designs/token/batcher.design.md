# token/batcher architecture design (draft)

this document defines token/batcher. it sanitizes token inputs and emits a `token::batch`.

## role
- convert raw token inputs into a canonical token batch.
- provide a single entrypoint for generator token batching.

## responsibilities
- validate token ids, counts, and array bounds.
- enforce sequence continuity and coupling constraints.
- normalize output selection defaults (`output_all` / `output_mask`).
- populate missing seq ids, seq masks, and positions when allowed.

## events (draft)
- `event::batch` inputs: token ids + token count, optional seq masks/primary ids/positions,
  output selection fields + capacities, requested step size/count + split mode (policy input).
- `events::batch_done` outputs: `token::batch` (sanitized token ids + metadata pointers + counts).
- `events::batch_error` outputs: error_out.

## batch semantics
- `token::batch` is a canonical, validated view of token ids and metadata.
