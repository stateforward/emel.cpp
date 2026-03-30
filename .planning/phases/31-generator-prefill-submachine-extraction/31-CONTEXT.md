---
phase: 31
slug: generator-prefill-submachine-extraction
created: 2026-03-29
status: ready
---

# Phase 31 Context

## Phase Boundary

Extract prefill orchestration into an explicit `src/emel/generator/prefill` machine while keeping
the generator domain and public generator boundary unchanged.

## Implementation Decisions

### Child Boundary
- The new child machine is generator-owned and synchronous; no queueing or deferred dispatch.
- The parent generator should own one explicit prefill actor boundary instead of inlining slots,
  snapshot, contract resolution, and compute dispatch in its top-level table.
- The child machine must consume a typed internal event and return through normal RTC completion,
  not by mutating parent state through ad hoc helpers.

### Ownership
- Parent generator remains the public actor and session owner.
- Child-machine runtime state should stay request-scoped on the internal event or the existing
  generate runtime context, not on new generator-context phase fields.
- Parent context may hold the child-machine interface, but prefill orchestration flags stay out of
  parent context.

### Guardrails
- No decode extraction in this phase.
- No `sm_any` attention-family split in this phase.
- No new implicit behavior; route choice remains explicit inside the child machine.

## Existing Code Insights

### Phase 30 Output
- `src/emel/generator/sm.hpp` already collapsed prefill into an explicit contract/result shape.
- The old prefill request and runtime guards are now a coherent unit that can move into a child
  machine without also dragging decode along.

### Reusable Patterns
- `emel::sm<context>` supports explicit child-machine construction with injected parent-owned
  context.
- Existing repo composition patterns keep the parent public wrapper and route child work through a
  small dispatch action rather than exposing child API surface directly.

## Specific Ideas

- Add `src/emel/generator/prefill/{sm,actions,guards,context,detail}.hpp`.
- Give the parent generator one `prefill_running` entry state plus one explicit result decision.
- Move all prefill-only actions and guards out of the parent generator files once the child is
  wired.
