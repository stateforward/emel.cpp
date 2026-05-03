---
phase: 192
slug: loader-tensor-outcome-contract
status: planned
gathered: 2026-05-03
---

# Phase 192 Context

## Audit Gap

The milestone audit found that the maintained loader tensor-load path still collapsed tensor actor
callbacks into a dispatch-local `tensor_load_capture` and then routed the loader state machine
through `load_ctx.err`.

## Source Context

- `src/emel/model/loader/sm.hpp` owns the top-level model-load orchestration.
- `src/emel/model/loader/actions.hpp` dispatches tensor-owned `bind_storage`, `plan_load`, and
  `apply_effect_results`.
- `src/emel/model/tensor` already exposes explicit done/error events for each tensor bulk phase.
- `tests/model/loader/lifecycle_tests.cpp` is the focused loader behavior and rule-regression test
  surface.

## Constraints

- Tensor bulk outcomes must route through explicit loader states and guards, not local callback flag
  aggregation.
- `load_ctx.err` may publish the final loader error only after explicit error-state selection.
- Tensor callbacks may synchronously write same-RTC result payloads but must not call
  `process_event`.
- The loader must continue to dispatch tensor actor events through the public tensor actor surface.
