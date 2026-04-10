---
phase: 42
slug: planner-event-boundaries
created: 2026-04-05
status: ready
---

# Phase 42 Context

## Phase Boundary

Phase 42 hard-cuts planner-family handoff so the top-level planner and the `simple`, `equal`, and
`sequential` child machines communicate through explicit typed machine events and explicit outcome
events instead of sharing the parent `request_runtime` type as the hidden cross-machine boundary.

This phase is not the place to finish the broad AGENTS prefix rename or to remove the per-dispatch
planner context. Those rule-compliance changes remain Phase 43 scope.

## Implementation Decisions

### Handoff Contract
- Keep the maintained batching algorithms intact.
- Introduce explicit per-mode request and outcome event types under each mode `events.hpp`.
- Give each mode machine its own wrapper `process_event(...)` surface so planner-to-mode dispatch is
  traceable through wrapper APIs rather than direct reach-through to child `model` state names.

### Parent Orchestration
- Keep the top-level planner state graph behaviorally equivalent where possible.
- Make the planner react to typed mode outcome completion events rather than assuming child
  completion of `planner::event::request_runtime`.
- Keep public planner request and callback-facing `plan_done` / `plan_error` events small and
  immutable.

### Verification Stance
- Add planner-family tests that prove the typed mode event surfaces exist and are wired through the
  wrappers.
- Re-run focused planner verification and the full repository quality gate after the contract
  change.

## Existing Code Insights

### Current Gaps
- `src/emel/batch/planner/sm.hpp` still enters `modes::<mode>::model` directly and branches on
  planner-owned `request_runtime` completion after child termination.
- All three mode `events.hpp` surfaces currently alias the parent planner event namespace instead of
  defining mode-local request/outcome contracts.
- Mode wrappers expose only the inherited generic `process_event(...)` path with no mode-owned
  request wrapper.

### Reusable Assets
- Phase 41 already established canonical mode-local `events.hpp` surfaces, so Phase 42 can fill
  them with real contract types rather than add more files.
- The parent planner already owns the callback-facing public `plan_done` / `plan_error` contract,
  which can stay unchanged while mode-local handoff becomes explicit.

## Specific Ideas

- Define mode-local request events that carry references to the planner request and request context.
- Define mode-local `_done` and `_error` outcome events so the parent can reason about mode results
  through explicit typed events.
- Add thin wrapper helpers in each mode `sm` that accept the mode-local request event and publish
  the corresponding mode-local outcome event.

## Deferred Ideas

- Full `state_` / `guard_` / `effect_` naming compliance remains Phase 43.
- Removing per-dispatch state from planner context remains Phase 43.
- Milestone-level behavior-preservation proof remains Phase 44.
