---
phase: 141
plan: 01
status: complete
requirements-completed:
  - TEXTGEN-04
  - TEXTGEN-05
---

# Phase 141 Summary

Phase 141 closed the remaining SML/detail/test-surface gaps from the v1.17 source-backed audit.

## Changes

- Added a kernel-owned `row_storage_bytes_for_dtype(...)` helper covering native f32, packed q8,
  packed q4, packed q6, prepared q6, and fallback quantized row sizing; generator row-storage
  sizing now delegates through it.
- Added `graph_lifecycle_snapshot` plus `event::capture_graph_lifecycle` for synchronous public
  graph lifecycle diagnostics.
- Added generator SML rows and a public `process_event(...)` wrapper for the lifecycle snapshot
  event, with runtime tensor availability modeled through explicit guards/transitions rather than
  action-local branching.
- Removed the direct `sm::graph_reservation()` and `sm::try_capture_graph_tensor(...)` accessors.
- Updated text-generator lifecycle tests to capture graph lifecycle data through the new event.
- Added source-level generator test-surface classification: maintained behavior proof is public
  lifecycle/parity/benchmark driven, while `action_guard_tests.cpp` and `detail_tests.cpp` are
  explicitly component-private regression coverage.

## Evidence

- Focused generator/runtime tests passed.
- Focused generator/runtime tests passed after the guard correction.
- `scripts/check_domain_boundaries.sh` passed.
- The scoped changed-file quality gate passed, including coverage, paritychecker tests, generation
  compare benchmark, docs generation, and boundary checks.
- Source scans found no maintained generation parity/benchmark actor-internal bridges and no
  remaining graph reservation/tensor capture context-reading `sm` wrappers.
