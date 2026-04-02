---
phase: 34
slug: lfm2-model-contract-bring-up
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 34 — Validation Strategy

## Quick Feedback Lane

- `rg -n "lfm2|shortconv|context_length" src/emel/model/data.cpp tests/model/loader/lifecycle_tests.cpp`
- `./build/zig/emel_tests_bin --test-case='model_execution_contract_rejects_lfm2_attention_block_with_shortconv_weights' --no-breaks`

## Full Verification

- `scripts/quality_gates.sh`

## Notes

- Phase 34 validation is tied to explicit `lfm2` contract truth in `src/emel` and focused loader
  coverage for maintained Liquid rejection/acceptance behavior.
