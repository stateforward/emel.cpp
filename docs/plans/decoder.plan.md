# Decoder parity plan (rolling)

This document tracks agreed decisions and upcoming work items for decoder parity with reference
implementations. Entries are appended as decisions are made.

## Decisions
- Memory decomposition will use `memory/coordinator/{recurrent,kv,hybrid}` under shared coordinator
  events and a common dispatch interface. See: encoder.
- Sampling output bridge will live in `decoder/sample_router` (seq-to-row mapping, backend sampling
  copy, per-sequence output constraint, per-row counts/buffers).
- Batch sanitizer will be shared under `emel/batch/sanitizer` (used by decoder and encoder).
- Output handling will be modeled internally in `decoder/output.hpp` (output capacity validation,
  output ordering, and reorder/swap tracking).
- Pooling will be handled via `decoder/pooling.hpp` (constexpr array of pooler structs/functions),
  not a separate SM.
- Graph scheduling will be shared under `graph/scheduler`, with decoder/encoder wrappers renamed to
  `decoder/graph_scheduler` and `encoder/graph_scheduler` (rename from compute_*).
- Batch sanitizer must ship with a dedicated `tools/bench` case and snapshot coverage.

## Decoder architecture
### Flow (top-level)
1. Sanitize/auto-generate batch inputs.
2. Split batch into ubatches and compute output selection counts.
3. Update memory coordinator (prepare update, optional optimize).
4. Prepare memory batch and KV cache for the full decode.
5. Reserve output buffers and reset output mappings.
6. Execute ubatches (compute + KV apply + output extraction).
7. Finalize outputs, reorder if needed, dispatch done/error.

### Components and responsibilities
- `decoder/sm.hpp`: Orchestrates the full decode flow and error handling.
- `decoder/ubatch_executor`: Per-ubatch orchestration; owns KV apply + compute executor execution.
- `decoder/compute_executor`: Graph prepare/alloc/bind/run/extract callbacks.
- `batch/sanitizer`: Validates batch fields, fills missing seq/pos/logits metadata, enforces
  continuity rules, and normalizes masks.
- `batch/splitter`: Splits sanitized batch into ubatches and computes output selection counts.
- `memory/coordinator/*`: Manages memory context lifecycle, update/optimize semantics, and status
  translation for decode orchestration.
- `kv/cache`: Owns slot planning, apply, rollback, and sequence operations.
- `decoder/output.hpp`: Output buffer reservation, output id mapping, reorder/swap tracking.
- `decoder/pooling.hpp`: Pooled embedding behaviors via constexpr pooler table.
- `decoder/sample_router`: Sampling output routing and backend sampling buffer copies.
- `decoder/graph_scheduler`: Shared graph scheduling wrapper (decoder-specific integration layer).

### Output and sampling surfaces
- Output buffer orchestration is internal to decoder; external callers provide capacity and receive
  output mappings and counts.
- Output reorder is applied lazily via `decoder/output.hpp` swap tracking and an explicit reorder
  helper.
- Sampling outputs are routed via `decoder/sample_router`, enforcing per-sequence output limits and
  mapping seq ids to output rows.

### Batch sanitization rules
- Auto-generate missing `seq_id`, `n_seq_id`, and positional data when not provided.
- Enforce sequence continuity and disallow partial sequence subsets within a decode step.
- Normalize output selection (output_all, output_mask, last-token default).

### Memory coordination semantics
- Use coordinator status to drive retryability and error mapping.
- One-shot optimize retry on prepare failure.
- Roll back KV and memory state on ubatch failure.

## Pending
- Final naming for cross-attention bridge (if needed).
- Decide how decoder exposes output buffer reservation to callers (helper API vs external owner).
- Decide how decoder exposes output reorder helper to callers.
