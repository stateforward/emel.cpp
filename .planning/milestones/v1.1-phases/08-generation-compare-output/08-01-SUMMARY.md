---
phase: 08-generation-compare-output
plan: 01
subsystem: generation-compare-contract
tags: [generation, benchmark, compare, contract]
requires:
  - phase: 07.1-replace-the-reference-backed-decode-path-with-native-emel-decode-before-any-further-benchmark-work
    provides: truthful native EMEL benchmark path and explicit reference path
provides:
  - explicit canonical generation case identity shared by the compare runner and generation bench
  - compare-mode failures for duplicate case names and missing canonical generation rows
  - a stable generation compare row in the normal `bench_runner --mode=compare` surface
affects: [08-02 published compare evidence, 09-benchmark-integration-hardening]
tech-stack:
  added: []
  patterns: [shared case-name constant, compare contract hardening, fail-fast pairing checks]
key-files:
  created: []
  modified: [tools/bench/bench_cases.hpp, tools/bench/bench_main.cpp]
key-decisions:
  - "Promote the canonical generation case name into `bench_cases.hpp` so compare-mode checks and benchmark registration share one identifier."
  - "Fail early on duplicate benchmark names and missing canonical generation rows instead of relying on sorted-position pairing alone."
  - "Keep the printed compare row format unchanged so Phase 8 hardens truth without creating Phase 9 snapshot churn."
patterns-established:
  - "Compare-mode contract checks can stay tool-local in `bench_main.cpp` while preserving the existing `emel.cpp ... llama.cpp ... ratio=...` output."
  - "Canonical-case presence is now enforced before ratio printing, so silent loss of the generation row is treated as an execution failure."
requirements-completed: [COMP-01]
duration: 0min
completed: 2026-03-10
---

# Phase 08 Plan 1 Summary

**The bench compare runner now enforces the canonical generation pairing contract explicitly**

## Accomplishments

- Updated `tools/bench/bench_cases.hpp` to expose the canonical generation case name once, so the
  generation benchmark registration and compare-mode validation use the same identifier.
- Hardened `tools/bench/bench_main.cpp` so compare mode now fails explicitly on duplicate EMEL or
  reference case names, missing canonical generation rows on either side, and case-count drift
  before printing ratios.
- Kept the normal compare output format unchanged, so the canonical generation row still appears as
  `emel.cpp ... llama.cpp ... ratio=...` while the pairing contract is now stricter underneath it.

## Task Commits

None. Execution stayed local with `commit_docs` disabled.

## Verification

- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare`
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare | rg "^generation/"`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`

## Deviations from Plan

- The hardening did not require edits to `tools/bench/generation_bench.cpp`; sharing the canonical
  case identifier through `bench_cases.hpp` was enough to keep the runner and generation case in
  sync.

## Next Readiness

- Phase 08 can move to published compare evidence in `08-02`.
- The normal compare flow now has an explicit canonical generation contract underneath it, so Wave
  2 can focus on the published compare surface instead of pairing ambiguity.

---
*Phase: 08-generation-compare-output*
*Completed: 2026-03-10*
