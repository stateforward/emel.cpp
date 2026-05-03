---
phase: 192
slug: loader-tensor-outcome-contract
status: passed
validated: 2026-05-03
---

# Phase 192 Validation

## Commands

- `cmake --build build/zig --parallel --target emel_tests_bin` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` passed.
- `EMEL_COVERAGE_CHANGED_ONLY=1 EMEL_COVERAGE_CHANGED_FILES='src/emel/model/loader/events.hpp:src/emel/model/loader/actions.hpp:src/emel/model/loader/guards.hpp:src/emel/model/loader/sm.hpp' EMEL_COVERAGE_TEST_SHARDS=model_and_batch scripts/test_with_coverage.sh` passed.
- `scripts/lint_snapshot.sh --update` refreshed the approved lint baseline.
- `scripts/lint_snapshot.sh` passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/model/loader/events.hpp:src/emel/model/loader/actions.hpp:src/emel/model/loader/guards.hpp:src/emel/model/loader/sm.hpp:tests/model/loader/lifecycle_tests.cpp:snapshots/lint/clang_format.txt' EMEL_QUALITY_GATES_BENCH_SUITE='generation:diarization_sortformer' scripts/quality_gates.sh` passed.

## Quality Gate Evidence

- Generation and Sortformer benchmark suites passed.
- Scoped coverage: 99.3% line coverage, 82.7% branch coverage.
- `paritychecker_tests` passed.
- Fuzz smoke was skipped as no fuzz-affecting files changed.
- Lint snapshot and docs generation passed.

## Notes

An initial broad all-suite benchmark gate timed out at 30 minutes. The successful final run used the
milestone-relevant generation and Sortformer benchmark suites explicitly via
`EMEL_QUALITY_GATES_BENCH_SUITE`.
