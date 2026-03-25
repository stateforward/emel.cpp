---
phase: 21-benchmark-attribution-and-impact
plan: 01
subsystem: benchmark-quantized-attribution
tags: [bench, generation, attribution, quantized, docs]
requires:
  - phase: 20-runtime-integration-and-proof
    provides: generator-level q2/q3/q6 optimized/shared attribution accessors
provides:
  - canonical benchmark publication of q2/q3/q6 optimized/shared attribution
  - compare-mode validation that rejects false AArch64 quantized-path claims
  - generated benchmark docs that surface quantized attribution alongside flash evidence
affects: [21-02 refreshed benchmark baselines]
tech-stack:
  added: []
  patterns: [canonical evidence state, compare-mode proof, generated docs publication]
key-files:
  created: []
  modified:
    [tools/bench/generation_bench.cpp, tools/bench/bench_main.cpp, tools/docsgen/docsgen.cpp]
key-decisions:
  - "Reuse the canonical `max_tokens=1` generation benchmark case as the maintained attribution proof surface instead of adding a second benchmark fixture."
  - "Keep AArch64 quantized proof strict in compare mode: optimized q2/q3/q6 counts must be non-zero and shared counts must stay zero."
requirements-completed: [BENCH-08]
duration: 0min
completed: 2026-03-23
---

# Phase 21 Plan 1 Summary

**The maintained benchmark compare path now publishes quantized attribution**

## Accomplishments

- Extended
  [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp)
  so the canonical generation benchmark evidence captures q2/q3/q6 optimized/shared dispatch
  counts alongside the existing flash evidence.
- Extended
  [bench_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/bench_main.cpp)
  so compare mode validates and prints a new `generation_quantized_evidence` line for the
  canonical maintained ARM workload.
- Extended
  [docsgen.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/docsgen/docsgen.cpp)
  so the generated benchmarks page surfaces the new quantized evidence instead of silently lagging
  the compare snapshot.

## Verification

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=compare | sed -n '1,6p'`
- `build/docsgen/docsgen --root . --check`

## Deviations from Plan

- None in scope. Attribution stayed on the maintained benchmark and docs publication surfaces.

---
*Phase: 21-benchmark-attribution-and-impact*
*Completed: 2026-03-23*
