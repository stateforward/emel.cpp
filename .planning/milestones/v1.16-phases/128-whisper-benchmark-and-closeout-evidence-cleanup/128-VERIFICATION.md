---
phase: 128
status: passed
verified: 2026-04-28
requirements: []
---

# Phase 128 Verification

## Verdict

**Passed.** The default Whisper benchmark closeout path now uses the stable 20-iteration sample
with explicit process-wall tolerance, while the contradiction tests still hard-fail the benchmark
for material evidence failures.

## Source Checks

- `scripts/bench_whisper_single_thread.sh` defaults
  `EMEL_WHISPER_BENCH_ITERATIONS` to 20 and forwards `--iterations "$ITERATIONS"` into the
  benchmark driver.
- `tools/bench/whisper_benchmark.py` defaults `--iterations` to 20 and publishes
  `performance_tolerance_ppm` in the benchmark summary.
- Phase 122 and Phase 125 closeout artifacts now include supersession notices that preserve
  historical evidence while pointing active closeout truth to the later source-backed chain.

## Build And Tests

- `cmake --build build/whisper_compare_tools --target whisper_benchmark_tests -j 6` passed.
- `build/whisper_compare_tools/whisper_benchmark_tests --no-breaks` passed: 12 test cases,
  168 assertions.
- Existing benchmark tests still cover transcript mismatch, model mismatch, reference lane
  failure, missing transcript, invalid warmups, invalid iterations, material slower EMEL means,
  deterministic reference policy, and recognizer-backed runtime metadata.

## Benchmark Evidence

Command:

```bash
scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build
```

Result:

- `benchmark_status=ok`
- `reason=ok`
- `iteration_count=20`
- `warmup_count=1`
- `performance_tolerance_ppm=20000`
- EMEL transcript `[C]`
- Reference transcript `[C]`
- EMEL mean `60,189,787 ns`
- Reference mean `60,736,881 ns`

## Diff Check

- `git diff --check -- scripts/bench_whisper_single_thread.sh tools/bench/whisper_benchmark.py tools/bench/whisper_benchmark_tests.cpp .planning/phases/122-whisper-final-gap-closeout-rerun/122-01-SUMMARY.md .planning/phases/122-whisper-final-gap-closeout-rerun/122-VERIFICATION.md .planning/phases/125-whisper-final-recognizer-closeout-rerun/125-01-SUMMARY.md .planning/phases/125-whisper-final-recognizer-closeout-rerun/125-VERIFICATION.md .planning/phases/128-whisper-benchmark-and-closeout-evidence-cleanup/128-CONTEXT.md .planning/phases/128-whisper-benchmark-and-closeout-evidence-cleanup/128-01-PLAN.md`
  exited 0.

## Requirement Results

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Active v1.16 requirements | unchanged | Phase 128 is tech-debt cleanup only; Phase 127 remains the active source-backed closeout truth. |
| Benchmark evidence stability | passed | The default closeout path uses 20 measured iterations and explicit 20,000 ppm process-wall tolerance. |
| Benchmark contradiction checks | passed | The focused benchmark test suite still passes all hard-fail regression cases. |
| Historical artifact truth | passed | Phase 122 and Phase 125 artifacts now state they are superseded for final closeout. |
