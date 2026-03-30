---
phase: 29
slug: qwen3-benchmark-publication
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-28
---

# Phase 29 — Validation Strategy

## Quick Feedback Lane

- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 EMEL_BENCH_GENERATION_ITERS=1 EMEL_BENCH_GENERATION_RUNS=1 EMEL_BENCH_GENERATION_WARMUP_ITERS=0 EMEL_BENCH_GENERATION_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only`

## Full Verification

- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 EMEL_BENCH_GENERATION_ITERS=1 EMEL_BENCH_GENERATION_RUNS=1 EMEL_BENCH_GENERATION_WARMUP_ITERS=0 EMEL_BENCH_GENERATION_WARMUP_RUNS=0 scripts/bench.sh --compare-update`
- `EMEL_BENCH_ITERS=1000 EMEL_BENCH_RUNS=3 EMEL_BENCH_WARMUP_ITERS=100 EMEL_BENCH_WARMUP_RUNS=1 EMEL_BENCH_GENERATION_ITERS=1 EMEL_BENCH_GENERATION_RUNS=1 EMEL_BENCH_GENERATION_WARMUP_ITERS=0 EMEL_BENCH_GENERATION_WARMUP_RUNS=0 scripts/bench.sh --snapshot --update`
- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh`

## Notes

- Snapshot refresh is already approved by the user for this run.
- Stored benchmark artifacts now publish explicit `benchmark_config` metadata alongside formatter,
  runtime, and native `q8_0` publication evidence for the canonical Qwen slice.
- `scripts/quality_gates.sh` completed after the benchmark baseline refresh; its internal compare
  lane prints explicit `benchmark_config` metadata and continues to use the repo gate's default
  generation run count.
