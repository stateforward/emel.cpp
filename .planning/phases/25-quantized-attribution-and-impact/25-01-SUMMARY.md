---
phase: 25-quantized-attribution-and-impact
plan: 01
subsystem: benchmark-runtime-contract-attribution
tags: [bench, quantized, runtime, attribution, compare]
requires:
  - phase: 24-quantized-path-proof-and-regression
    provides: maintained `8/4/0/0` runtime-contract proof on the shipped generator surface
provides:
  - benchmark-time publication of the shipped runtime contract counts
  - canonical compare-mode hard-fail validation of the approved `8/4/0/0` contract
  - docs-generation support for the new runtime-contract metadata
affects: [25-02 stored benchmark snapshot and docs publication]
tech-stack:
  added: []
  patterns:
    [benchmark runtime-contract attribution, additive wrapper accessors, compare hard-fail validation]
key-files:
  created: []
  modified:
    [tools/bench/generation_bench.cpp, tools/bench/bench_main.cpp, tools/docsgen/docsgen.cpp]
key-decisions:
  - "Publish benchmark-time runtime-contract truth from the shipped generator wrapper instead of inferring it from tool-local counters."
  - "Keep the approved dense-f32-by-contract seams explicit in compare/docs publication rather than calling the supported path fully quantized everywhere."
requirements-completed: [BENCH-10]
duration: 0min
completed: 2026-03-25
---

# Phase 25 Plan 1 Summary

**Benchmark compare output now publishes and validates the shipped runtime contract**

## Accomplishments

- Extended
  [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp)
  so canonical benchmark capture records the shipped generator runtime-contract counts alongside
  the existing flash and quantized dispatch evidence.
- Hardened
  [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp)
  so compare mode now prints `generation_runtime_contract:` and fails if the canonical benchmark
  case drifts away from the approved `native_quantized=8`,
  `approved_dense_f32_by_contract=4`, `disallowed_fallback=0`, and `explicit_no_claim=0`
  contract.
- Updated
  [docsgen.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/docsgen/docsgen.cpp)
  to parse and publish the new runtime-contract metadata plus an honest contract-summary sentence,
  while still stopping short of any stored snapshot or docs refresh before approval.

## Verification

- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 scripts/bench.sh --compare`

## Deviations from Plan

- Minor scope expansion only: `tools/docsgen/docsgen.cpp` was updated in the implementation step
  so generated docs could understand the new benchmark metadata once approval for artifact refresh
  was granted. No stored snapshot or generated docs artifact was modified before approval.

---
*Phase: 25-quantized-attribution-and-impact*
*Completed: 2026-03-25*
