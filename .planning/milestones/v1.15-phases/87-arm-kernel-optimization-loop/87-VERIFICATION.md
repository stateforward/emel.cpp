# Phase 87 Verification

## Commands

- `git diff --check -- src/emel/diarization/sortformer/detail.cpp .planning/phases/87-arm-kernel-optimization-loop`
  - Result: passed.
- `cmake --build build/coverage --target emel_tests_bin -j 6`
  - Result: passed.
- `cmake --build build/bench_tools_ninja --target bench_runner -j 6`
  - Result: passed.
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
  - Result: passed, 1/1 tests.
- `EMEL_BENCH_SUITE=diarization_sortformer EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=compare`
  - Result: passed.
  - Before/after sample: Phase 86 evidence had `end_to_end_ns=2871626` and
    `transformer_ns=2554333`; Phase 87 evidence had `end_to_end_ns=2652040` and
    `transformer_ns=2398083`.
- `scripts/quality_gates.sh`
  - Result: exited 0.
  - Coverage: changed-source line coverage 96.6%, branch coverage 68.5%.
  - Notes: benchmark step reported expected unsnapshotted entries for the two
    `diarization/sortformer` benchmark rows; the quality gate script continued and printed
    `warning: benchmark snapshot regression ignored by quality gates`.

## Success Criteria

1. Each optimization starts from Phase 86 evidence and records before/after timing.
   - Covered by the dense helper before/after stage profile evidence.
2. Reusable operator work lands in the correct ownership layer.
   - The optimized helper remains Sortformer-local because this code path currently has no general
     kernel tensor contract; the remaining broader dense/matmul kernelization candidate is recorded.
3. No disallowed fallback, reference dependency, or runtime dispatcher was introduced.
   - Covered by the implementation: no new fallback path, no reference calls, and no dispatch table.
4. Loop continues until no justified material candidate remains.
   - Covered by the stop condition: the next material candidate requires a broader kernel contract,
     not another low-risk local loop item.
