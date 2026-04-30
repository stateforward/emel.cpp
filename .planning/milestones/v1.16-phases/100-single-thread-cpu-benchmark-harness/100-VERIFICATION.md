---
phase: 100-single-thread-cpu-benchmark-harness
verified: 2026-04-26T16:04:53Z
status: passed
score: 4/4 must-haves verified
---

# Phase 100: Single-Thread CPU Benchmark Harness Verification Report

**Phase Goal:** Publish CPU-only single-thread EMEL and `whisper.cpp` benchmark records on the same
ARM host.
**Verified:** 2026-04-26T16:04:53Z
**Status:** passed

## Goal Achievement

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Both lanes run with exactly one CPU thread and record that constraint. | VERIFIED | `scripts/bench_whisper_single_thread.sh` invokes `tools/bench/whisper_benchmark.py`; reference uses `--threads 1 --processors 1 --no-gpu`; records contain `thread_count=1`, `processor_count=1`, and `cpu_only=true`. |
| 2 | Benchmarks use the same pinned model variant and audio fixture as parity. | VERIFIED | Records use comparison group `whisper/tiny/q8_0/phase99_440hz_16khz_mono`; model/audio SHA256s match Phase 99 evidence. |
| 3 | Records include warmup count, iteration count, wall time, stage timings, host identity, and backend version/build identity. | VERIFIED | `whisper_benchmark/v1` raw records include warmups, iterations, wall times, EMEL/reference stage timings, host identity, EMEL git build identity, and `whisper.cpp` tag/commit. |
| 4 | Generated docs distinguish benchmark proof from parity proof. | VERIFIED | Phase 100 summary and verification explicitly reference `whisper_benchmark/v1` and defer Phase 99 `bounded_drift` parity interpretation. |

**Score:** 4/4 must-haves verified

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| BENCH-01 | SATISFIED | CPU-only single-thread EMEL and `whisper.cpp` lanes run on the same ARM host with matching model/audio fixtures. |
| BENCH-02 | SATISFIED | Benchmark records label backend, thread count, ARM host, fixture identity, warmup/iteration counts, and measured stage timings. |

## Automated Checks

- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build --warmups 1 --iterations 3` -
  passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="<Phase 100 files>" EMEL_QUALITY_GATES_BENCH_SUITE=whisper_single_thread scripts/quality_gates.sh` -
  passed.

## Human Verification Required

None.

## Residual Notes

- Phase 101 should treat the benchmark record as the baseline for profiling and optimization.
- The reference benchmark lane is CPU-only despite the reference build probing GPU capability,
  because the invocation passes `--no-gpu`.
