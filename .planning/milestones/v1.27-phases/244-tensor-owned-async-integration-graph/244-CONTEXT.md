# Phase 244: Tensor-Owned Async Integration Graph - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Integrate cooperative async loading through the tensor actor while preserving tensor-owned residency.
This phase adds a direct tensor-owned async load request surface. Maintained loader/tool entrypoints
remain deferred to Phase 245.

</domain>

<decisions>
## Implementation Decisions

### Ownership
- `model/tensor` owns residency state and commits resident lifecycle only on terminal async success.
- `emel/io/async` owns byte-window progress behavior only; it does not mutate tensor residency.
- Tensor dispatches async work only via `emel::io::async::sm::process_event(...)`.

### Outcomes
- Tensor publishes explicit progress, done, and error outcomes for async load requests.
- Partial progress leaves tensor lifecycle unbound.
- Terminal success sets tensor lifecycle resident and records the target buffer/window bytes.

</decisions>

<code_context>
## Existing Code Insights

- `request_read_load` and `request_staged_load` already provide tensor-owned direct strategy
  dispatch patterns.
- `io/async` now exposes bounded progress and terminal outcomes through public callbacks.

</code_context>

<specifics>
## Specific Ideas

Tests should bind tensor metadata, dispatch async progress twice through `model/tensor`, inspect
state after partial and terminal progress, and verify invalid/error paths through public dispatch.

</specifics>

<deferred>
## Deferred Ideas

Loader strategy selection, maintained entrypoints, and cross-strategy benchmark publication are
deferred to Phases 245 and 247.

</deferred>
