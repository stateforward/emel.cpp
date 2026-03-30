---
phase: 32
slug: prefill-surface-shrink-and-proof
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-29
---

# Phase 32 — Validation Strategy

## Quick Feedback Lane

- `build/zig/emel_tests_bin --test-case='*generator_prefill*'`
- `build/zig/emel_tests_bin --test-case='*generator_sm_*'`
- `build/zig/emel_tests_bin --test-case='*qwen3*generator*'`

## Full Verification

- `build/paritychecker_zig/paritychecker_tests --test-case='*qwen3*generation*'`
- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh`
- `EMEL_BENCH_GENERATION_ITERS=1 EMEL_BENCH_GENERATION_RUNS=1 EMEL_BENCH_GENERATION_WARMUP_ITERS=0 EMEL_BENCH_GENERATION_WARMUP_RUNS=0 scripts/bench.sh --compare --generation-only`

## Notes

- The performance check is part of the done condition for this milestone because the user
  explicitly asked for the performance impact of the decomposition work.
- Focused generator/prefill topology tests, maintained Qwen generator tests, maintained Qwen
  paritychecker generation tests, and the maintained generation-only compare lane all passed after
  the Phase 32 shrink.
- `scripts/quality_gates.sh` reran but did not close cleanly because the broad benchmark snapshot
  gate reported unrelated regressions in `text/encoders/*` and `tokenizer/full_wpm_*`, not in the
  maintained generator/prefill surfaces touched by this milestone.
- The user explicitly waived those unrelated broad benchmark snapshot regressions for this phase,
  so Phase 32 closes on the maintained generator/parity/compare proof rather than on the noisy
  out-of-scope benchmark rows.
