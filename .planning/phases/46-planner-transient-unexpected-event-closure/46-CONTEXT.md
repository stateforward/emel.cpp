---
phase: 46
slug: planner-transient-unexpected-event-closure
created: 2026-04-05
status: ready
---

# Phase 46 Context

## Phase Boundary

Phase 46 closes the remaining v1.10 milestone audit gap without widening scope beyond the planner
family.

The latest audit reports one actionable gap from `.planning/v1.10-MILESTONE-AUDIT.md`:
- FINDING-01: `state_simple_mode`, `state_equal_mode`, and `state_sequential_mode` in
  `src/emel/batch/planner/sm.hpp` lack explicit `unexpected_event` handling even though the states
  are synchronous-transient and the audit downgraded runtime risk to non-blocking

No requirement, flow, or broader integration rewiring gap remains. This is a focused structural
closeout phase.

## Implementation Decisions

### Scope
- Keep all runtime changes inside `src/emel/batch/planner` and focused planner-family tests.
- Do not widen into generator families, snapshot refresh, benchmark work, or broad repository
  cleanup.

### Closure Strategy
- Prefer a planner-owned fix that makes unexpected-event behavior explicit at the affected
  transient states.
- If the architecture intentionally relies on a transient-state exemption, document and encode that
  decision in planner-owned surfaces so the audit no longer reports silent dropping.
- Preserve the existing run-to-completion synchronous mode-dispatch behavior.

### Proof Expectations
- Add a focused failing test first if runtime behavior changes are required to close the finding.
- Re-audit the milestone after the phase completes to verify FINDING-01 no longer appears under
  `gaps.integration`.

## Existing Code Insights

### Current Audit Evidence
- `.planning/v1.10-MILESTONE-AUDIT.md` marks FINDING-01 as non-blocking because no external event
  can normally arrive while the machine is in the transient mode execution states.
- The affected states are `state_simple_mode`, `state_equal_mode`, and `state_sequential_mode` in
  `src/emel/batch/planner/sm.hpp`.

### Relevant Proof Surfaces
- `tests/batch/planner/planner_sm_transition_tests.cpp` already exercises top-level planner
  transition behavior.
- `tests/batch/planner/planner_sm_flow_tests.cpp` covers repeated dispatch and recovery flows that
  should remain unchanged by any closure.

## Deferred Ideas

- Refreshing `snapshots/lint/clang_format.txt` remains deferred until the user explicitly approves
  snapshot updates.
- Nyquist backfill remains a separate follow-up from this milestone-gap closure phase.
