---
phase: 41
slug: planner-mode-surface-cutover
created: 2026-04-05
status: ready
---

# Phase 41 Context

## Phase Boundary

Phase 41 hard-cuts only the planner child-mode surface so `simple`, `sequential`, and `equal`
live under `src/emel/batch/planner/modes/` with canonical file bases and no leftover shared
legacy mode surface.

This phase is not the place to redesign planner-to-mode handoff, rename planner-family outcome
events, or change how per-dispatch state flows through the planner family. Those behavior-boundary
changes stay in phases 42 and 43.

## Implementation Decisions

### Infrastructure Phase
- This is an infrastructure-only cutover. The agent may choose the smallest structural change that
  leaves each mode with canonical `actions`, `context`, `detail`, `errors`, `events`, `guards`,
  and `sm` files.
- Shared helper logic may remain shared, but it must move behind a canonical planner-owned surface
  instead of the leftover `src/emel/batch/planner/modes/detail.hpp` root helper.

### Surface Discipline
- Keep the behavioral algorithms for `simple`, `sequential`, and `equal` intact in this phase.
- Avoid changing planner-to-mode dispatch structure or event payload semantics.
- Prefer alias or forwarding surfaces where that preserves behavior and avoids duplicated helper
  logic.

### Verification Stance
- Focus verification on planner-mode compile surface and maintained planner tests.
- Required validation for this phase is the focused planner test slice plus the repo quality gate.

## Existing Code Insights

### Current Gaps
- `src/emel/batch/planner/modes/` currently exposes only `actions`, `guards`, and `sm` per mode.
- A leftover shared `src/emel/batch/planner/modes/detail.hpp` helper surface is still imported by
  planner and mode files.
- The current test file `tests/batch/planner/modes/detail_tests.cpp` is attached to that shared
  helper surface rather than a canonical planner or mode family surface.

### Reusable Assets
- Top-level planner already owns canonical `context`, `errors`, `events`, `guards`, and `sm`
  surfaces.
- Existing mode tests under `tests/batch/planner/modes/` already exercise `simple`,
  `sequential`, and `equal` behavior, so structural aliasing can be verified without changing
  maintained batching semantics.

## Specific Ideas

- Promote the shared helper implementation to `src/emel/batch/planner/detail.hpp`, then give each
  mode its own canonical wrapper files under `modes/<mode>/`.
- Replace the shared mode-detail test with a planner-family detail test that follows the new helper
  ownership.

## Deferred Ideas

- Planner-to-mode typed event hardening remains phase 42.
- Transition-form and persistent-state compliance cleanup remains phase 43.
- Milestone-level behavior-preservation proof remains phase 44.
