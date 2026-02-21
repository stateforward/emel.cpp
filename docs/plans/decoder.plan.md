# decoder parity plan (rolling)

this document tracks agreed decisions and upcoming work items for decoder parity with reference
implementations. entries are appended as decisions are made.

## decisions
- memory decomposition will use `memory/coordinator/{recurrent,kv,hybrid}` under shared coordinator
  events and a common dispatch interface. see: encoder.
- sampling output bridge will live in `decoder/sample_router` (seq-to-row mapping, backend sampling
  copy, per-sequence output constraint, per-row counts/buffers).
- batch sanitizer will be shared under `emel/batch/sanitizer` (used by decoder and encoder).
- output handling will be modeled internally in `decoder/output.hpp` (output capacity validation,
  output ordering, and reorder/swap tracking).
- pooling will be handled via `decoder/pooling.hpp` (constexpr array of pooler structs/functions),
  not a separate SM.
- graph scheduling will be shared under `graph/scheduler`, with decoder/encoder wrappers renamed to
  `decoder/graph_scheduler` and `encoder/graph_scheduler` (rename from compute_*).
- batch sanitizer must ship with a dedicated `tools/bench` case and snapshot coverage.

## decoder architecture
### flow (top-level)
1. sanitize/auto-generate batch inputs.
2. split batch into ubatches and compute output selection counts.
3. update memory coordinator (prepare update, optional optimize).
4. prepare memory batch and KV cache for the full decode.
5. reserve output buffers and reset output mappings.
6. execute ubatches (compute + KV apply + output extraction).
7. finalize outputs, reorder if needed, dispatch done/error.

### components and responsibilities
- `decoder/sm.hpp`: orchestrates the full decode flow and error handling.
- `decoder/ubatch_executor`: per-ubatch orchestration; owns KV apply + compute executor execution.
- `decoder/compute_executor`: graph prepare/alloc/bind/run/extract callbacks.
- `batch/sanitizer`: validates batch fields, fills missing seq/pos/logits metadata, enforces
  continuity rules, and normalizes masks.
- `batch/splitter`: splits sanitized batch into ubatches and computes output selection counts.
- `memory/coordinator/*`: manages memory context lifecycle, update/optimize semantics, and status
  translation for decode orchestration.
- `kv/cache`: owns slot planning, apply, rollback, and sequence operations.
- `decoder/output.hpp`: output buffer reservation, output id mapping, reorder/swap tracking.
- `decoder/pooling.hpp`: pooled embedding behaviors via constexpr pooler table.
- `decoder/sample_router`: sampling output routing and backend sampling buffer copies.
- `decoder/graph_scheduler`: shared graph scheduling wrapper (decoder-specific integration layer).

### output and sampling surfaces
- output buffer orchestration is internal to decoder; external callers provide capacity and receive
  output mappings and counts.
- output reorder is applied lazily via `decoder/output.hpp` swap tracking and an explicit reorder
  helper.
- sampling outputs are routed via `decoder/sample_router`, enforcing per-sequence output limits and
  mapping seq ids to output rows.

### batch sanitization rules
- auto-generate missing `seq_id`, `n_seq_id`, and positional data when not provided.
- enforce sequence continuity and disallow partial sequence subsets within a decode step.
- normalize output selection (output_all, output_mask, last-token default).

### memory coordination semantics
- use coordinator status to drive retryability and error mapping.
- one-shot optimize retry on prepare failure.
- roll back KV and memory state on ubatch failure.

## pending
- final naming for cross-attention bridge (if needed).
- decide how decoder exposes output buffer reservation to callers (helper API vs external owner).
- decide how decoder exposes output reorder helper to callers.
