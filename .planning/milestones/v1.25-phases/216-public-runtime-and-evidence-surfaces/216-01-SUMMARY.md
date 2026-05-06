---
phase: 216-public-runtime-and-evidence-surfaces
plan: 01
status: complete
type: summary
completed: 2026-05-05T18:27:09Z
requirements:
  - TIO-03
  - VAL-04
---

# Phase 216 Summary

## Implemented

- Added public `used_io_strategy` evidence to model-loader `load_done` and
  requested/used I/O strategy evidence to `load_error`.
- Marked the strategy as used only when the model-loader RTC reaches the explicit
  `io_load_done_all` transition.
- Added `tools/bench/model_load_strategy.hpp` so maintained tools select strategy
  through `EMEL_MODEL_LOAD_IO_STRATEGY` and bind a public `io::loader::sm` to the
  model-loader request.
- Updated generation benchmark, Sortformer benchmark fixture, embedded probe, and
  paritychecker to report load strategy from public model-loader events instead of
  callback-time tensor actor probes.
- Added model-loader tests for read/copy success, unsupported strategy reporting,
  and maintained tool source guardrails.

## Evidence

- Generation, diarization, paritychecker, and embedded probe tool builds compile
  through the new public helper.
- Tool compare tests for generation, diarization, and paritychecker pass.
- Generated model-loader architecture docs now show the `effect_mark_io_strategy_used`
  transition from `state_io_load_decision` to `state_tensor_apply_dispatch`.

## Deferred

Phase 217 owns broader behavior guardrails and naming clarity around the read/copy
strategy versus the out-of-scope staged/chunked constrained-memory policy.
