---
phase: 36
slug: parity-and-regression-proof
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 36 — Validation Strategy

## Quick Feedback Lane

- `ls snapshots/parity | rg 'generation_lfm2_5_1_2b_thinking_q4_k_m'`
- `rg -n "maintained generation|supported fixtures|append-only" tools/paritychecker/paritychecker_tests.cpp tools/generation_fixture_registry.hpp`

## Full Verification

- `scripts/quality_gates.sh`

## Notes

- Phase 36 validation is additive: the maintained Liquid parity surface is checked alongside the
  preserved maintained fixture registry and parity tests.
