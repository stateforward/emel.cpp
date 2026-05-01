---
phase: 128
plan: 01
status: complete
completed: 2026-04-28
requirements-completed: []
---

# Phase 128 Summary: Whisper Benchmark And Closeout Evidence Cleanup

## Outcome

Phase 128 closed the non-blocking audit debt around noisy closeout benchmark defaults and stale
historical closeout prose. Active v1.16 requirements remain complete through Phase 127.

## Completed Work

- Changed the default Whisper single-thread closeout sample to 20 measured iterations in both
  `scripts/bench_whisper_single_thread.sh` and `tools/bench/whisper_benchmark.py`.
- Added an explicit `--performance-tolerance-ppm` benchmark setting with a 20,000 ppm default so
  sub-percent process-wall jitter does not become false closeout debt.
- Preserved hard failures for material performance regressions, transcript drift, model mismatch,
  reference lane failures, missing transcripts, invalid warmups, and invalid iterations.
- Added benchmark tests proving the Python driver and shell wrapper defaults use the stable
  closeout sample.
- Marked Phase 122 and Phase 125 closeout artifacts as superseded by Phase 126, Phase 127, and
  the latest source-backed audit chain.

## Evidence

- `cmake --build build/whisper_compare_tools --target whisper_benchmark_tests -j 6` passed.
- `build/whisper_compare_tools/whisper_benchmark_tests --no-breaks` passed: 12 test cases,
  168 assertions.
- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` passed with
  `benchmark_status=ok reason=ok`.
- `build/whisper_benchmark/benchmark_summary.json` records 20 iterations, one warmup, tolerance
  `20000`, exact `[C]` transcripts on both lanes, EMEL mean `60,189,787 ns`, and reference mean
  `60,736,881 ns`.
- `git diff --check` passed for the Phase 128 touched files.

## Notes

This phase does not change the active requirement owner for `CLOSE-01`. Phase 127 remains the
active closeout truth; Phase 128 only stabilizes evidence collection and repairs superseded
ledger wording.
