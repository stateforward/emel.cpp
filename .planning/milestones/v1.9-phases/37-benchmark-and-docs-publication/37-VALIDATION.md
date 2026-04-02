---
phase: 37
slug: benchmark-and-docs-publication
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 37 — Validation Strategy

## Quick Feedback Lane

- `rg -n "lfm2_5_1_2b_thinking_q4_k_m|generation_formatter_contract|generation_stage_probe" snapshots/bench/benchmarks_compare.txt docs/benchmarks.md`
- `./build/bench_tools_ninja/bench_runner_tests --test-case='bench_runner generation compare keeps maintained Qwen and Liquid fixtures' --no-breaks`

## Full Verification

- `scripts/quality_gates.sh`

## Notes

- Phase 37 validation is tied to the stored compare/docs publication surface and the maintained
  benchmark-runner regression test.
