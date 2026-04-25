---
phase: 90
plan: 01
status: complete
completed: 2026-04-23
requirements-completed:
  - PRF-01
  - PRF-02
  - BEN-01
  - DOC-01
  - RUN-03
  - OUT-03
---

# Phase 90 Plan 1 Summary: Runtime Parity And Benchmark Truth Repair

## Completed Work

- Replaced the Sortformer parity test's fabricated probability fill with a maintained
  `sortformer/pipeline::sm` run from canonical PCM through the Phase 89 runtime orchestrator.
- Added a shared non-`src/` Sortformer benchmark/test fixture so parity and benchmark harnesses use
  the same canonical in-memory model contract without duplicating setup logic.
- Reworked `tools/bench/diarization/sortformer_bench.cpp` so the EMEL lane measures
  `pipeline::sm::process_event(...)` and publishes pipeline metadata/checksum evidence instead of
  calling output helpers directly.
- Kept the reference lane separate as a recorded segment baseline with no shared EMEL runtime,
  cache, model, or actor state.
- Updated `docs/templates/benchmarks.md.j2` and generated `docs/benchmarks.md` to state the
  pipeline-backed EMEL lane, recorded-baseline limitation, checksum, and non-claim around external
  reference execution.

## Evidence

- Canonical pipeline output checksum: `13736986938186292950`.
- Canonical output shape: 3 segment records over the maintained `188 x 4` probability output.
- Sortformer compare metadata now reports workload `diarization_sortformer_pipeline_v1`.
- Stage-profile publication now reports workload `diarization_sortformer_pipeline_profile_v1`.

## Notes

The benchmark case clamps Sortformer runtime measurement to one orchestrated pipeline pass so the
full quality gate remains practical now that the EMEL lane executes the real maintained pipeline.
