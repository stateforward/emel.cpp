---
phase: 42-planner-event-boundaries
plan: 01
subsystem: batch
tags: [planner, modes, sml, events, wrappers, agents]
requires:
  - phase: 41-planner-mode-surface-cutover
    provides: canonical planner-mode file surfaces
provides:
  - explicit mode-local request and outcome event contracts
  - planner-to-mode wrapper dispatch through mode-owned process_event surfaces
  - planner tests proving typed mode outcome emission
affects: [43-planner-rule-compliance, 44-behavior-preservation-proof]
tech-stack:
  added: []
  patterns: [typed same-RTC wrapper dispatch, explicit mode-local outcome events]
key-files:
  created:
    - .planning/phases/42-planner-event-boundaries/42-CONTEXT.md
    - .planning/phases/42-planner-event-boundaries/42-01-PLAN.md
  modified:
    - src/emel/batch/planner/actions.hpp
    - src/emel/batch/planner/sm.hpp
    - src/emel/batch/planner/modes/simple/events.hpp
    - src/emel/batch/planner/modes/simple/sm.hpp
    - src/emel/batch/planner/modes/equal/events.hpp
    - src/emel/batch/planner/modes/equal/sm.hpp
    - src/emel/batch/planner/modes/sequential/events.hpp
    - src/emel/batch/planner/modes/sequential/sm.hpp
    - tests/batch/planner/planner_surface_tests.cpp
key-decisions:
  - "Kept the public planner request and callback-facing plan_done/plan_error contract unchanged while adding mode-local internal request and outcome event types."
  - "Moved planner-to-mode dispatch out of embedded child model states and into explicit wrapper-dispatch effects so the handoff is readable through mode-owned process_event surfaces."
  - "Implemented the mode wrappers as standalone orchestrators over existing mode guards/actions because the previous child models were only valid as embedded completion-driven submachines."
patterns-established:
  - "Planner-family child machines may expose mode-local typed request and _done/_error events while still reusing shared planner runtime helpers internally."
  - "A standalone wrapper surface may own the synchronous guard/action sequence when a legacy child SML graph depended on parent submachine completion semantics."
requirements-completed: [MODE-02, RULE-02]
duration: 67min
completed: 2026-04-05
commit: pending
---

# Phase 42: Planner Event Boundaries Summary

**Planner-to-mode handoff now goes through explicit mode-local request/outcome events and
mode-owned wrapper dispatch, instead of the parent planner entering child `model` states and
reasoning about hidden `request_runtime` completion.**

## Performance

- **Duration:** 67 min
- **Started:** 2026-04-05T05:24:06Z
- **Completed:** 2026-04-05T06:31:00Z
- **Tasks:** 4
- **Files modified:** 11

## Accomplishments

- Added explicit mode-local `event::request` plus `events::plan_done` / `events::plan_error`
  contracts for `simple`, `equal`, and `sequential`.
- Switched the top-level planner from embedded child-model mode states to explicit wrapper-dispatch
  effects and planner-local mode states.
- Gave each mode wrapper a real standalone `process_event(...)` implementation that reuses the
  existing mode guards/actions and emits typed outcome events for the parent handoff.
- Extended planner surface tests to lock the new event-boundary contract and proved the rebuilt
  planner test slice still passes.

## Task Commits

Each task was applied in the working tree without a commit in this session:

1. **Task 1: Define mode-local request and outcome events** - `pending`
2. **Task 2: Replace embedded planner-mode reach-through with wrapper dispatch** - `pending`
3. **Task 3: Add event-boundary surface tests** - `pending`
4. **Task 4: Rebuild and verify the planner family through the full gate** - `pending`

## Decisions Made

- The parent planner now chooses behavior in `sm.hpp`, but the synchronous mode handoff is
  executed through `action::effect_plan_with_*_mode` wrapper effects instead of embedded child
  state-machine states.
- Mode wrappers emit typed outcome events while still sharing the planner-owned request/runtime
  storage for same-RTC internal execution. That keeps the public planner event surface unchanged
  while making the internal handoff explicit.
- Standalone wrapper orchestration reuses the existing mode actions and guards rather than trying to
  preserve the old parent-dependent child completion semantics directly.

## Issues Encountered

- The first wrapper implementation tried to drive the child mode `sm` directly as though it were a
  standalone state machine. That failed because the old child graphs were only valid when entered
  as embedded submachines from parent completion edges.
- `scripts/quality_gates.sh` reported benchmark snapshot regressions for
  `logits/validator_sml/*`, but the script ended with
  `warning: benchmark snapshot regression ignored by quality gates`, so the gate still passed.

## Next Phase Readiness

- Phase 43 can now focus on rule-compliance cleanup: destination-first row style, AGENTS prefix
  naming, and removing per-dispatch runtime state from planner context.
- Phase 44 can use the explicit wrapper boundary to prove behavior preservation without relying on
  hidden child completion semantics.
