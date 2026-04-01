---
phase: 33
slug: generator-initializer-submachine-extraction
created: 2026-03-31
status: ready
---

# Phase 33 Context

## Phase Boundary

Extract initialize orchestration into an explicit `src/emel/generator/initializer` machine while
keeping the generator public actor, validation entrypoints, and initialize publication behavior in
the parent.

## Implementation Decisions

### Child Boundary
- The child owns `begin_initialize`, conditioner bind, renderer initialize, memory reserve,
  optional graph reserve, and sampling-mode configuration.
- The parent keeps public `initialize_run` validation, backend-ready short-circuiting in the C++
  wrapper, and done/error channel publication.
- Parent-to-child handoff stays same-RTC through a typed internal `initializer::event::run`.

### Ownership
- The parent generator remains the session owner and stores the child actor handle in generator
  context.
- Request-scoped data stays on `event::initialize_ctx`; no new generator-context phase flags are
  introduced.
- Child actions mutate only injected parent-owned generator context through the child action
  context reference.

### Guardrails
- No decode extraction in this milestone slice.
- No new public API surface.
- No queueing, deferred events, or mailbox behavior.

## Existing Code Insights

### Reusable Pattern
- `src/emel/generator/prefill` already demonstrates the target composition style:
  parent-owned child actor, typed internal run event, and explicit result classification.

### Parent Seam
- `src/emel/generator/sm.hpp` has a contiguous initialize block that can collapse to one
  `initializing -> initializer_result_decision` seam without touching generate/decode behavior.

## Specific Ideas

- Add `src/emel/generator/initializer/{sm,actions,guards,context,detail}.hpp`.
- Add one focused initializer topology test file under `tests/generator/initializer/`.
- Update parent lifecycle topology assertions to prove the child boundary is explicit.
