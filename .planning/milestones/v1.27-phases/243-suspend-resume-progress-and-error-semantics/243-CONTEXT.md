# Phase 243: Suspend/Resume Progress and Error Semantics - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement bounded cooperative progress for the async I/O strategy. This phase adds explicit partial
progress and terminal success through public dispatch. It does not yet integrate tensor residency or
loader selection; those land in later phases.

</domain>

<decisions>
## Implementation Decisions

### Progress Model
- Use caller-owned source and target windows in async storage.
- A public `load_window` dispatch advances at most one configured chunk.
- Repeated dispatch with the same caller-owned progress storage is the cooperative resume tick.

### Outcome Model
- Publish `load_window_progress_done` for partial progress.
- Publish `load_window_done` only when the logical byte span is complete.
- Publish deterministic `_error` outcomes for validation/source/progress/cancel paths.

### Control Flow
- Runtime choices stay in guards and `sm.hpp` states: valid vs invalid, cancelled vs active, partial
  vs terminal.
- Actions only perform bounded byte movement or publish the already-selected outcome.

</decisions>

<code_context>
## Existing Code Insights

- Phase 242 already added caller-owned `load_window_storage` and `load_window_progress`.
- `io/staged_read` provides a source-span copy reference, but Phase 243 uses one bounded chunk per
  public async dispatch instead of a full-span loop.

</code_context>

<specifics>
## Specific Ideas

Tests should drive `emel::io::async::sm` directly through public `process_event(...)`, proving
partial progress, deterministic resume ordering, terminal success, cancellation, and validation
errors.

</specifics>

<deferred>
## Deferred Ideas

Tensor-owned state integration and maintained loader entrypoints are deferred to Phases 244-245.
Benchmark publication and cross-strategy performance comparison are deferred to Phase 247.

</deferred>
