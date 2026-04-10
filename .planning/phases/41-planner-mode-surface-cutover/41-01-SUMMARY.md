---
phase: 41-planner-mode-surface-cutover
plan: 01
subsystem: batch
tags: [planner, modes, sml, surface, agents]
requires:
  - phase: 40-planner-surface-cutover
    provides: canonical top-level planner surface and wrapper naming
provides:
  - canonical per-mode file surfaces for simple, sequential, and equal
  - planner-owned shared helper surface at src/emel/batch/planner/detail.hpp
  - planner-family tests retargeted off the deleted modes/detail.hpp surface
affects: [42-planner-event-boundaries, 43-planner-rule-compliance, 44-behavior-preservation-proof]
tech-stack:
  added: []
  patterns: [canonical per-mode surface wrappers, planner-owned shared detail]
key-files:
  created:
    - src/emel/batch/planner/detail.hpp
    - src/emel/batch/planner/modes/simple/context.hpp
    - src/emel/batch/planner/modes/simple/detail.hpp
    - src/emel/batch/planner/modes/simple/errors.hpp
    - src/emel/batch/planner/modes/simple/events.hpp
    - src/emel/batch/planner/modes/sequential/context.hpp
    - src/emel/batch/planner/modes/sequential/detail.hpp
    - src/emel/batch/planner/modes/sequential/errors.hpp
    - src/emel/batch/planner/modes/sequential/events.hpp
    - src/emel/batch/planner/modes/equal/context.hpp
    - src/emel/batch/planner/modes/equal/detail.hpp
    - src/emel/batch/planner/modes/equal/errors.hpp
    - src/emel/batch/planner/modes/equal/events.hpp
    - tests/batch/planner/planner_detail_tests.cpp
  modified:
    - src/emel/batch/planner/actions.hpp
    - src/emel/batch/planner/guards.hpp
    - src/emel/batch/planner/modes/simple/actions.hpp
    - src/emel/batch/planner/modes/simple/guards.hpp
    - src/emel/batch/planner/modes/sequential/actions.hpp
    - src/emel/batch/planner/modes/sequential/guards.hpp
    - src/emel/batch/planner/modes/equal/actions.hpp
    - src/emel/batch/planner/modes/equal/guards.hpp
    - tests/batch/planner/planner_surface_tests.cpp
    - CMakeLists.txt
key-decisions:
  - "Promoted the shared planner-mode helper surface from modes/detail.hpp to planner/detail.hpp so the shared implementation remains planner-owned rather than mode-root legacy."
  - "Used thin per-mode wrapper headers for context, detail, errors, and events so each mode directory exposes canonical file bases without duplicating runtime logic."
patterns-established:
  - "Planner-family shared helpers belong at the owning planner level, while mode directories expose only canonical family file bases."
  - "Surface-only cutovers may use alias wrappers when behavior preservation matters more than immediate naming/rule cleanup."
requirements-completed: [MODE-01, MODE-03]
duration: 24min
completed: 2026-04-05
commit: pending
---

# Phase 41: Planner Mode Surface Cutover Summary

**Planner modes now expose canonical per-mode file surfaces under `src/emel/batch/planner/modes/`, while the leftover shared `modes/detail.hpp` helper was replaced with a planner-owned `detail.hpp` surface and the planner test slice stayed green.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-04-05T04:54:39Z
- **Completed:** 2026-04-05T05:18:38Z
- **Tasks:** 3
- **Files modified:** 25

## Accomplishments
- Added canonical `context`, `detail`, `errors`, and `events` surfaces for `simple`,
  `sequential`, and `equal` so each mode directory now shows the allowed file bases locally.
- Moved the shared planner-mode helper implementation to
  `src/emel/batch/planner/detail.hpp` and deleted the leftover
  `src/emel/batch/planner/modes/detail.hpp` legacy surface.
- Retargeted planner tests and build entries to the new helper ownership and validated the planner
  doctest slice plus the full quality gate.

## Task Commits

Each task was applied in the working tree without a commit in this session:

1. **Task 1: Canonicalize shared planner-mode helper ownership** - `pending`
2. **Task 2: Add per-mode canonical wrapper surfaces** - `pending`
3. **Task 3: Retarget planner tests and verification** - `pending`

**Plan metadata:** `pending`

## Files Created/Modified
- `src/emel/batch/planner/detail.hpp` - Canonical planner-owned shared helper surface for planner
  and mode logic.
- `src/emel/batch/planner/modes/simple/*.hpp` - Canonical per-mode wrapper surfaces for the simple
  planner mode family.
- `src/emel/batch/planner/modes/sequential/*.hpp` - Canonical per-mode wrapper surfaces for the
  sequential planner mode family.
- `src/emel/batch/planner/modes/equal/*.hpp` - Canonical per-mode wrapper surfaces for the equal
  planner mode family.
- `tests/batch/planner/planner_detail_tests.cpp` - Shared planner helper tests retargeted to the
  canonical planner detail surface.
- `tests/batch/planner/planner_surface_tests.cpp` - Surface assertions extended to cover the new
  per-mode canonical aliases.

## Decisions Made
- Shared helper logic remains centralized, but its ownership is now truthful: planner-owned shared
  code lives in planner `detail.hpp`, not under a mode-root helper path.
- Per-mode wrapper headers are acceptable for Phase 41 because this phase is about canonical file
  surfaces, not suffix-renaming or behavior-boundary redesign.

## Deviations from Plan

None - plan executed as written.

## Issues Encountered

- `ctest --test-dir build/zig -R planner` does not match any registered test name in this repo
  because planner tests are sharded into `emel_tests_model_and_batch`. Verification used the
  equivalent focused doctest invocation
  `./build/zig/emel_tests_bin --no-breaks --source-file='*tests/batch/planner/*'` instead.
- The full `scripts/quality_gates.sh` run has a long `bench_runner --mode=compare` tail. The run
  completed successfully, but it required waiting for the compare bench rather than assuming the
  gate had stalled.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Planner mode directories now expose canonical surface files, so Phase 42 can focus on typed
  planner-to-mode event boundaries instead of layout cleanup.
- AGENTS naming prefixes such as `guard_` and `state_` are intentionally still outstanding; that
  rule-compliance rename work remains Phase 43 scope.
