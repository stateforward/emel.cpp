---
phase: 100-single-thread-cpu-benchmark-harness
plan: 01
requirements-completed: [BENCH-01, BENCH-02]
completed: 2026-04-26
---

# Phase 100 Plan 01: Single-Thread CPU Benchmark Harness - Execution Summary

**Phase Goal:** Publish CPU-only single-thread EMEL and `whisper.cpp` benchmark records on the same
ARM host.

**Status:** Complete for Phase 100 scope.

## Outcomes

### Benchmark Records

- Added `tools/bench/whisper_benchmark.py`.
- Added `scripts/bench_whisper_single_thread.sh`.
- Added `whisper_benchmark/v1` raw lane records and `whisper_benchmark_summary/v1` summary output.
- Records include host identity, model/audio checksums, backend identity, warmup count, iteration
  count, thread count, processor count, CPU-only flag, process wall time, transcript checksum, and
  lane status.

### Stage Timing

- Extended `tools/bench/whisper_emel_parity_runner.cpp` to emit EMEL stage timings:
  model load, audio load, GGUF binding, contract build, encode, decode, publish, and total wall
  time.
- The benchmark harness parses `whisper.cpp` timing output for load, mel, sample, encode, decode,
  batch decode, prompt, and reported total times.

### Invocation Constraints

- Reference lane invokes pinned `whisper.cpp` v1.7.6 with
  `--threads 1 --processors 1 --no-gpu`.
- EMEL lane remains a separate process using EMEL-owned loader/runtime actors only.

## Benchmark Evidence

Latest local summary from `build/whisper_benchmark/benchmark_summary.json`:

| Lane | Iterations | Mean process wall time | Min | Max | Transcript |
|------|------------|------------------------|-----|-----|------------|
| EMEL | 3 | 327,299,736 ns | 325,549,292 ns | 330,234,708 ns | `token:50257` |
| whisper.cpp | 3 | 419,345,402 ns | 416,157,458 ns | 423,989,667 ns | `[Bell]` |

Host: `Darwin arm64` (`macOS-15.1-arm64-arm-64bit-Mach-O`).

## Verification Commands

- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build --warmups 1 --iterations 3` -
  passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES="<Phase 100 files>" EMEL_QUALITY_GATES_BENCH_SUITE=whisper_single_thread scripts/quality_gates.sh` -
  passed.

## Stored Evidence

- Summary: `build/whisper_benchmark/benchmark_summary.json`
- Raw EMEL records: `build/whisper_benchmark/raw/emel_benchmark.jsonl`
- Raw reference records: `build/whisper_benchmark/raw/reference_benchmark.jsonl`

## Notes

- The benchmark shows EMEL faster on process wall time for this maintained local benchmark shape.
  Phase 101 still owns profiling and optimization closure.
- The benchmark proof is separate from Phase 99 parity proof; transcript drift remains recorded as
  `bounded_drift` in the parity summary.
