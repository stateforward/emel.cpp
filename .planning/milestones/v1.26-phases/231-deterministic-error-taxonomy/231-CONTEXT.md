# Phase 231: Deterministic Error Taxonomy - Context

**Gathered:** 2026-05-07  
**Status:** In progress

<domain>
## Phase boundary

Phase 231 lands deterministic staged-read error taxonomy for current source-backed
`io/staged_read` behavior only:

- **ESG-01:** pre-I/O guard failures map to named deterministic categories.
- **ESG-02A:** source-contract read-surface failures map to named deterministic categories
  (`null_source_span`, `source_span_size_mismatch`, `insufficient_source_span`).
- **ESG-03:** staged sequencing/stage-contract failures map to named deterministic categories.
- **ESG-04:** actor/API boundary remains exception-free.

`ESG-02B` is explicitly deferred: file open/seek/read/per-stage short-read categories
require a future approved file-backed staged source path that owns handles/syscalls.

</domain>

<decisions>
## Implementation decisions

- Keep `staged_read` source-span-only (no OS I/O widening, no synthetic fault knobs).
- Model new categories through explicit guards/transitions/actions in `sm.hpp`.
- Keep runtime behavior choice in guards/transitions only; no routing in actions/detail.
- Verify categories only through public `process_event(...)` doctests.

</decisions>

<code_context>
## Existing code insights

- `src/emel/io/staged_read/events.hpp` exposes source-span requests (`source_span`,
  `source_span_bytes`) with no file handle/path.
- `src/emel/io/staged_read/actions.hpp` executes in-memory copy via `memcpy` only.
- `src/emel/io/staged_read/errors.hpp` is the category surface consumed by callbacks/tests.

</code_context>

<deferred>
## Deferred items

- ESG-02B file open/seek/read + per-stage short-read categories.
- Any staged actor file descriptor/handle ownership and lifetime semantics.

</deferred>
