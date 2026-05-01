# Phase 100: Single-Thread CPU Benchmark Harness - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Publish CPU-only, single-thread benchmark records for the same pinned Whisper tiny q8_0 model/audio
pair used by Phase 99 parity. This phase owns timing metadata and benchmark record publication,
not parity verdict interpretation or optimization work.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Contract
- Use `whisper_benchmark/v1` raw lane records.
- Use `whisper_benchmark_summary/v1` for the machine-readable summary.
- Record `thread_count=1`, `processor_count=1`, `cpu_only=true`, warmup count, iteration count,
  host identity, model/audio SHA256s, backend identity, and backend build identity.

### Lane Execution
- Reuse the Phase 99 reference setup and EMEL runner.
- Invoke `whisper.cpp` with `--threads 1 --processors 1 --no-gpu`.
- Keep EMEL in its isolated process using EMEL loader, encoder actor, and decoder actor.
- Measure process wall time in the Python harness and stage timings where the lane exposes them.

</decisions>

<code_context>
## Integration Points

- `tools/bench/whisper_emel_parity_runner.cpp` now emits EMEL stage timings for model load, audio
  load, binding, contract build, encode, decode, publish, and total wall time.
- `tools/bench/whisper_benchmark.py` runs warmups and measured iterations for both lanes and writes
  raw JSONL plus a summary.
- `scripts/bench_whisper_single_thread.sh` builds/reuses pinned reference and EMEL tools and runs
  the benchmark harness.
- `scripts/quality_gates.sh` recognizes `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_single_thread`.

</code_context>

<specifics>
## Maintained Benchmark

Comparison group:

`whisper/tiny/q8_0/phase99_440hz_16khz_mono`

Default local benchmark shape:

- Warmups: 1
- Iterations: 3
- Reference: pinned `whisper.cpp` v1.7.6 commit
  `a8d002cfd879315632a579e73f0148d06959de36`
- Host from evidence: Darwin arm64

</specifics>

<deferred>
## Deferred Ideas

- Profiling individual EMEL bottlenecks and landing optimizations belong to Phase 101.
- Closeout-wide full gate and source-backed milestone ledger belong to Phase 102.

</deferred>
