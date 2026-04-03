---
phase: 35
slug: maintained-runtime-execution-on-arm
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 35 — Validation Strategy

## Quick Feedback Lane

- `ls snapshots/parity | rg 'generation_lfm2_5_1_2b_thinking_q4_k_m'`
- `rg -n "generation_runtime_contract|generation_quantized_evidence|lfm2" docs/benchmarks.md tools/bench/generation_bench.cpp tools/paritychecker/parity_runner.cpp`

## Full Verification

- `scripts/quality_gates.sh`

## Notes

- Phase 35 validation is based on the maintained Liquid runtime evidence already published on the
  parity and benchmark surfaces, plus the green full-repo gate for the current branch.
