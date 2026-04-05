---
phase: 40-planner-surface-cutover
plan: 01
completed: 2026-04-04
commit: pending
---

# Phase 40 Plan 01 Summary

The canonical planner entrypoint now exports an additive top-level `emel::BatchPlanner` alias from
`src/emel/batch/planner/sm.hpp`, the legacy `BatchSplitter` umbrella alias has been replaced with
`BatchPlanner`, and a focused planner surface test now locks that naming contract into the batch
planner test suite.
