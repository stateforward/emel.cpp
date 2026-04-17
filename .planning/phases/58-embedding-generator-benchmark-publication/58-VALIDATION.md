---
phase: 58
slug: embedding-generator-benchmark-publication
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-16
---

# Phase 58 — Validation Strategy

## Quick Feedback Lane

- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 ./build/bench_tools_ninja/embedding_generator_bench_runner`

## Full Verification

- `cmake --build build/bench_tools_ninja --target embedding_generator_bench_runner -j4`
- `EMEL_BENCH_ITERS=10 EMEL_BENCH_RUNS=5 ./build/bench_tools_ninja/embedding_generator_bench_runner`

## Notes

- Validation is satisfied when the maintained benchmark drives real TE embedding requests through
  `src/emel/embeddings/generator` and reports the pinned fixture/contract metadata.
