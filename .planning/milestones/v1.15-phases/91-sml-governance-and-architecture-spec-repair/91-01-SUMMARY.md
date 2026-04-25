---
phase: 91
plan: 01
status: complete
completed: 2026-04-23
requirements-completed:
  - RUN-02
  - DIA-03
  - DOC-01
---

# Phase 91 Plan 1 Summary: SML Governance And Architecture Spec Repair

## Completed Work

- Replaced optional `error_out` branching in diarization request and Sortformer executor publication
  with explicit `guard_has_error_out` / `guard_no_error_out` decision states.
- Replaced executor transformer hidden-buffer lane selection with compile-time-selected layer
  execution helpers so runtime lane choice no longer occurs inside the action body.
- Moved generated machine documentation from `docs/architecture/` to `.planning/architecture/`.
- Updated docsgen and README generation to reference `.planning/architecture/`, then removed the
  stale generated `docs/architecture/` tree.

## Outcome

- Maintained diarization actors no longer use action-side runtime branching for optional sink
  publication or hidden-buffer lane selection.
- Generated machine documentation is preserved for planning/inspection without violating the
  repository rule against maintained machine-definition specs under `docs/architecture/*`.
