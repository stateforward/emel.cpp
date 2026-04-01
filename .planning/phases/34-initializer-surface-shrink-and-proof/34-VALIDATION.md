---
phase: 34
slug: initializer-surface-shrink-and-proof
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-31
---

# Phase 34 — Validation Strategy

## Quick Feedback Lane

- `build/zig/emel_tests_bin --test-case='*generator_sm_*'`
- `build/zig/emel_tests_bin --test-case='*qwen3*generator*'`
- `scripts/generate_docs.sh`

## Full Verification

- `build/paritychecker_zig/paritychecker_tests --test-case='*qwen3*generation*'`
- `EMEL_BENCH_GENERATION_ITERS=1 EMEL_BENCH_GENERATION_RUNS=1 EMEL_BENCH_GENERATION_WARMUP_ITERS=0 EMEL_BENCH_GENERATION_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only`
- `scripts/quality_gates.sh`

## Notes

- `build/zig/emel_tests_bin --test-case='*generator_sm_*'` passed.
- `build/zig/emel_tests_bin --test-case='*qwen3*generator*'` passed.
- `build/paritychecker_zig/paritychecker_tests --test-case='*qwen3*generation*'` passed.
- `scripts/generate_docs.sh` regenerated the parent generator docs and new
  `generator_initializer` docs.
- The generation-only compare lane stayed effectively flat to favorable:
  `max_tokens_1` `0.946x`, `10` `1.027x`, `100` `0.990x`, `1000` `0.899x` versus llama.cpp
  on the maintained Qwen case.
- `scripts/quality_gates.sh` passed. It emitted a warning about an ignored benchmark snapshot
  regression in `text/jinja/formatter_long`, which is outside the initializer slice.
