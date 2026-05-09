---
phase: 248
status: passed
requirements:
  - PERF-01
---

# Phase 248 Verification

## Result: Passed

`PERF-01` is satisfied. The maintained generation benchmark now accepts
`EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async` and executes the public async loader route instead
of failing with `io_strategy_unavailable`.

## Evidence

| Check | Result | Evidence |
|-------|--------|----------|
| Focused loader tests | pass | `./build/debug/emel_tests_bin --test-case="*io loader*" --test-case="*model loader*"`: 34 test cases, 683 assertions |
| Maintained benchmark | pass | `EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async scripts/bench.sh --snapshot --compare --suite=generation`: `488169958 ns/op` EMEL, `353585500 ns/op` reference, `1.381x` |
| Required tests/lint | pass | `ctest --test-dir build/debug -R "emel_tests|lint_snapshot" --output-on-failure`: 14/14 passed |
| Scoped quality gate | pass | `EMEL_QUALITY_GATES_BENCH_SUITE="generation" EMEL_QUALITY_GATES_CHANGED_FILES="..." scripts/quality_gates.sh`: exit 0, generation lane `425780125 ns/op` EMEL, `316074125 ns/op` reference, `1.347x` |

## Notes

The measured `cooperative_async` run is slower than the reference lane on this fixture. The phase
goal was not to claim an async speedup; it was to replace the unsupported maintained path with real
source-backed execution and honest measured evidence.

For constrained RAM and large models, the architectural benefit is that the maintained strategy now
uses bounded async window progress into request-owned tensor storage before `model/tensor` commits
residency. That preserves the path needed to avoid representing async loading as an all-at-once
payload materialization.
