---
phase: 43
slug: planner-rule-compliance
created: 2026-04-05
status: ready
---

# Phase 43 Context

## Phase Boundary

Phase 43 brings the planner family into the AGENTS hard-cut naming and rule contract without
widening milestone scope beyond `src/emel/batch/planner` and its child mode families.

The concrete gap left by Phase 42 is twofold:
- planner-family state, guard, and transition-effect symbols still use legacy unprefixed names
- mode wrapper `process_event(...)` functions still select runtime behavior in member functions
  instead of letting the owning SML machines drive that control flow

## Implementation Decisions

### Naming Contract
- Rename planner-family SML state symbols to `state_*`.
- Rename planner-family runtime predicates to `guard_*`.
- Rename planner-family transition effects to `effect_*`.
- Keep public event naming unchanged because Phase 42 already aligned request and `_done` /
  `_error` contracts.

### Wrapper Compliance
- Keep the explicit mode-local request and outcome event boundary from Phase 42.
- Remove handwritten runtime path selection from mode wrapper member functions.
- Have each mode wrapper translate the mode-local request to `planner_event::request_runtime`,
  drive its own `model`, then emit typed `_done` / `_error` from explicit machine outcome states.

### Persistent State
- Preserve the current empty planner-family `action::context` because the family still has no
  persistent actor-owned state.
- Keep per-dispatch data in `event::request_ctx` only.

## Existing Code Insights

### Current Gaps
- `src/emel/batch/planner/sm.hpp` still exposes legacy top-level state symbols such as
  `initialized`, `validate_decision`, `publishing`, and `done`.
- `src/emel/batch/planner/modes/*/sm.hpp` still exposes legacy mode state symbols and manual
  runtime branching in wrapper member functions.
- Planner-family tests still reference the old guard/effect names directly.

### Reusable Assets
- Phase 42 already proved the explicit mode request/outcome wrappers and planner/event boundary.
- Focused planner tests already cover both direct action/guard surfaces and wrapper event behavior,
  so Phase 43 can lock the rename and wrapper-control-flow changes without inventing new harnesses.

## Deferred Ideas

- Milestone-level batching-behavior proof remains Phase 44 scope.
- Any broader repo-wide AGENTS rename outside the planner family remains out of scope.
