---
phase: 117
plan: 01
status: complete
completed: 2026-04-27
requirements_completed:
  - REOPEN-01
---

# Phase 117 Summary

## Outcome

Maintained Whisper transcript drift now fails the compare wrapper and the `whisper_compare`
quality-gate lane. `exact_match` remains the only successful compare status; `bounded_drift` is
still published with `reason=transcript_mismatch` for diagnostics but exits nonzero.

## Changes

- Added focused doctest coverage for Whisper compare exact-match success and transcript-mismatch
  failure in `tools/bench/whisper_benchmark_tests.cpp`.
- Added `WHISPER_COMPARE_SCRIPT_PATH` to the Whisper bench test target.
- Updated `tools/bench/whisper_compare.py` to return success only for `exact_match`.
- Stabilized the existing fake-runner benchmark policy test by slowing the fake reference lane.

## Evidence

- `build/whisper_compare_tools/whisper_benchmark_tests`: 9 test cases, 130 assertions passed.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build`: exact match passed.
- Scoped quality gate passed with `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare`.
- `scripts/check_domain_boundaries.sh` passed.

## Requirement Impact

`REOPEN-01` is complete: `bounded_drift` transcript mismatch is not accepted by the maintained
compare gate.
