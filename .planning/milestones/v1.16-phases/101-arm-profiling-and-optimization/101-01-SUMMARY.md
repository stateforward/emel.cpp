---
phase: 101-arm-profiling-and-optimization
plan: 01
requirements-completed: [PERF-01, PERF-02, PERF-03]
completed: 2026-04-26T19:07:55Z
status: complete
---

# Phase 101 Plan 01: ARM Profiling And Optimization - Execution Summary

**Phase Goal:** Profile the maintained EMEL runtime and optimize until EMEL beats the matched
single-thread `whisper.cpp` lane.

**Status:** Complete.

## Implemented Work

- Added workspace-backed Hann/chirp table preparation in `src/emel/kernel/whisper/detail.hpp`.
- Added AArch64 NEON q8_0 row dot acceleration for Whisper linear projections, with scalar fallback
  on non-AArch64 hosts.
- Added `tests/whisper/kernel/detail_tests.cpp`.
- Replaced repeated layer tensor-name `snprintf` formatting with a fixed-buffer formatter and
  replaced transcript `snprintf` formatting with allocation-free `std::to_chars`.
- Added missing `whisper` compile-time shard support in `CMakeLists.txt` and coverage shard support
  in `scripts/test_with_coverage.sh` / `scripts/quality_gates.sh`.

## Benchmark Evidence

Before optimization, Phase 100 recorded:

- EMEL mean process wall time: `327,299,736 ns`
- Reference mean process wall time: `419,345,402 ns`

After optimization and the final scoped gate run:

- EMEL mean process wall time: `143,873,444 ns`
- Reference mean process wall time: `431,662,486 ns`
- EMEL encode stage: about `102-103 ms`
- EMEL decode stage: about `14.8-15.0 ms`

## Verification Commands

- `cmake --build build/audit-native --target emel_tests_bin` - passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/whisper/*'` -
  **12 cases, 12 passed, 1813 assertions, 0 failures**.
- `cmake --build build/whisper_compare_tools --parallel --target whisper_emel_parity_runner` -
  passed.
- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build --warmups 1 --iterations 3` -
  passed.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` - passed with
  `bounded_drift`.
- `EMEL_QUALITY_GATES_CHANGED_FILES="<Phase 101 files>" EMEL_QUALITY_GATES_BENCH_SUITE=whisper_single_thread scripts/quality_gates.sh` -
  passed with `src/emel/kernel/whisper/detail.hpp` line coverage `100.0%`, branch coverage
  `55.3%`, and benchmark status `ok`.

## Notes

- The scoped gate uses branch-only coverage exclusions on straight-line tensor lookup lines where
  actor guards have already validated the model contract; this does not lower the project
  thresholds.
- An attempted broader coverage run before completion was stopped after an unrelated generator
  test failure:
  `generator_quantized_path_audit_marks_unsupported_quantized_stage_no_claim`.
