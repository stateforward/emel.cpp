# Decoder parity plan (rolling)

This document tracks agreed decisions and upcoming work items for decoder parity with
`tmp/llama.cpp`. Entries are appended as decisions are made.

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

## Pending
- Final naming for cross-attention bridge (if needed).
