---
phase: 60
slug: reconcile-maintained-te-quant-scope-and-proof-truth
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-16
---

# Phase 60 — Validation Strategy

## Quick Feedback Lane

- `./build/coverage/emel_tests_bin --no-breaks --test-case='maintained TE fixture is documented in tests/models README,maintained TE q5 fixture matches locked local size when present,TE q5 proof compares EMEL outputs against stored upstream goldens,embeddings generator initializes with TE q5 fixture when present,maintained TE fixture selector approves q8 and q5 only'`

## Full Verification

- `cmake --build build/bench_tools_ninja --target embedding_generator_bench_runner -j4`
- `EMEL_TE_FIXTURE=tests/models/TE-75M-q5_0.gguf EMEL_BENCH_ITERS=10 EMEL_BENCH_RUNS=5 ./build/bench_tools_ninja/embedding_generator_bench_runner`

## Notes

- This phase is valid when q5 is explicitly documented and proved as an approved maintained TE
  slice, without widening support claims beyond q8/q5.
