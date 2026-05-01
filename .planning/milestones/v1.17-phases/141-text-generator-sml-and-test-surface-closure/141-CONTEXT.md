# Phase 141: Text Generator SML And Test Surface Closure - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning
**Mode:** Auto-generated during autonomous gap-closure execution

<domain>
## Phase Boundary

Close the v1.17 audit gaps for the moved text generator SML and test surface without changing
generation behavior, model scope, sampling policy, or benchmark claims.

</domain>

<decisions>
## Implementation Decisions

### Scope
- Preserve the existing maintained generation path and parity/benchmark behavior.
- Prefer event-driven public diagnostics over `sm` wrapper functions that read context directly.
- Move reusable dtype row-storage ownership into the kernel layer before wiring generator code to it.
- Do not treat passing tests alone as sufficient if source scans still show actor-internal coupling.

### the agent's Discretion
The implementation may leave broader generator decomposition for a later phase, but Phase 141
verification must clearly report any remaining SML/detail/test-surface gaps.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `emel::graph::event::capture_graph_lifecycle`-style public events can expose diagnostic state
  through synchronous RTC dispatch without adding stored request pointers.
- `src/emel/kernel/detail.hpp` already owns quantized dtype storage helpers that generator code can
  delegate to.

### Established Patterns
- Generator machine public wrappers call `process_event(...)` on immutable event payloads.
- Tests under `tests/text/generator/**` currently include a mix of machine-level lifecycle tests
  and lower-level action/detail helper tests.

### Integration Points
- Maintained parity/benchmark proof must continue through generator public events and must not
  include actor internals.

</code_context>

<specifics>
## Specific Ideas

Use a narrow event-driven snapshot to replace direct context-reading graph reservation and tensor
capture accessors.

</specifics>

<deferred>
## Deferred Ideas

Complete removal or reclassification of generator actor-internal action/detail tests remains needed.

</deferred>
