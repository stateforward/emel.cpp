---
phase: 45
slug: planner-audit-closeout
created: 2026-04-05
status: ready
---

# Phase 45 Context

## Phase Boundary

Phase 45 closes the remaining v1.10 audit gaps without widening scope beyond the planner family.

The phase has two required closure targets from `.planning/v1.10-MILESTONE-AUDIT.md`:
- Phase 40 proof is orphaned because `40-01-SUMMARY.md` lacks `requirements-completed` and Phase 40
  has no `40-VERIFICATION.md`
- the three planner mode wrapper `process_event(...)` methods still contain a runtime `if` that
  chooses `on_done` vs `on_error`, which the audit left as FINDING-02 against `RULE-01`

The stale lint snapshot finding is explicitly deferred because snapshot updates require separate
user consent under `AGENTS.md`.

## Implementation Decisions

### Audit Closure Scope
- Treat Phase 45 as both a narrow code-repair phase and a proof backfill phase.
- Keep all runtime changes inside `src/emel/batch/planner` and its focused planner tests.
- Do not widen into snapshot refresh or broader repository cleanup.

### Wrapper Compliance
- Remove the remaining runtime branch statement from each mode wrapper `process_event(...)`.
- Prefer explicit state inspection through `visit_current_states` / `is(...)` over handwritten
  runtime branch statements in wrapper member functions.
- Keep the existing typed mode `plan_done` / `plan_error` callback boundary intact.

### Proof Backfill
- Add a focused failing test first that reproduces the wrapper-rule violation.
- Backfill Phase 40 artifacts so PLAN-01, PLAN-02, and PLAN-03 are explicitly proven rather than
  only implied by code inspection.
- Re-run the milestone audit after Phase 45 closes to confirm only the explicitly deferred snapshot
  warning remains.

## Existing Code Insights

### Current Audit Gaps
- `src/emel/batch/planner/modes/simple/sm.hpp`, `equal/sm.hpp`, and `sequential/sm.hpp` still use
  `if (accepted && guard::guard_planning_succeeded(...))` inside wrapper member functions.
- `src/emel/batch/planner/actions.hpp` still captures mode callbacks through a helper that assumes
  wrappers publish done/error outcomes back into planner `request_ctx`.
- `.planning/phases/40-planner-surface-cutover/40-01-SUMMARY.md` exists but has no
  `requirements-completed` frontmatter, and `.planning/phases/40-planner-surface-cutover` has no
  verification report.

### Reusable Assets
- `tests/batch/planner/planner_surface_tests.cpp` already proves the canonical planner aliases and
  typed mode wrapper callback surface.
- `.planning/v1.10-MILESTONE-AUDIT.md` already contains the evidence needed to write the missing
  Phase 40 verification report once the artifact gap is closed.

## Deferred Ideas

- Refreshing `snapshots/lint/clang_format.txt` remains deferred until the user explicitly approves
  snapshot updates.
- Nyquist backfill remains a separate follow-up from milestone audit closeout.
