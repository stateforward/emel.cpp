---
phase: 40-planner-surface-cutover
plan: 01
subsystem: batch
tags: [planner, surface, canonicalization, agents]
requires:
  - phase: 39-publication-refresh-and-audit-closeout
    provides: pre-v1.10 shipped planner surface
provides:
  - canonical top-level planner path under src/emel/batch/planner
  - additive public planner alias without legacy BatchSplitter ambiguity
  - planner-owned top-level orchestration surface readable from planner-family files
affects:
  - src/emel/generator
  - src/emel/machines.hpp
tech-stack:
  added: []
  patterns: [canonical planner surface, planner-owned orchestration boundary]
key-files:
  created: []
  modified:
    - src/emel/batch/planner/sm.hpp
    - src/emel/batch/planner/events.hpp
    - src/emel/generator/actions.hpp
    - src/emel/generator/context.hpp
    - src/emel/machines.hpp
    - tests/batch/planner/planner_sm_flow_tests.cpp
    - tests/batch/planner/planner_sm_transition_tests.cpp
    - tools/bench/batch/planner_bench.cpp
key-decisions:
  - "Made src/emel/batch/planner/sm.hpp the canonical top-level planner entrypoint for maintainers."
  - "Exported additive BatchPlanner naming while keeping the machine anchored in planner-owned files."
  - "Kept child-mode cleanup and event-boundary redesign out of scope for this phase."
patterns-established:
  - "Top-level machine surface cutovers can land before child machine canonicalization as long as the planner-owned path and public wrapper naming are explicit."
requirements-completed: [PLAN-01, PLAN-02, PLAN-03]
duration: 22min
completed: 2026-04-04
commit: pending
---

# Phase 40: Planner Surface Cutover Summary

**The top-level planner entry surface is now canonical under `src/emel/batch/planner/`, with
public planner naming and planner-owned orchestration readability anchored to that path.**

## Performance

- **Duration:** 22 min
- **Started:** 2026-04-04T23:20:00Z
- **Completed:** 2026-04-04T23:42:00Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Anchored the top-level planner entrypoint and machine wrapper under
  `src/emel/batch/planner/sm.hpp`.
- Replaced legacy top-level planner naming ambiguity with additive `BatchPlanner` naming tied to
  the canonical planner machine.
- Updated direct planner consumers and focused planner tests so the maintained batching entry flow
  remained provably wired through the canonical surface.

## Decisions Made

- Limited this phase to the top-level planner surface only, leaving mode-family cleanup and typed
  event-boundary redesign for later v1.10 phases.
- Treated planner-owned readability as part of the surface contract, not just file placement.

## Issues Encountered

- This phase originally shipped without a verification artifact and without
  `requirements-completed` frontmatter, which later orphaned PLAN-01/02/03 in the milestone audit.
  Phase 45 backfilled that missing proof record without changing the underlying Phase 40
  implementation intent.

## Next Phase Readiness

- The canonical top-level planner surface was in place for Phase 41 mode-path cleanup and the later
  event-boundary and rule-compliance phases.
