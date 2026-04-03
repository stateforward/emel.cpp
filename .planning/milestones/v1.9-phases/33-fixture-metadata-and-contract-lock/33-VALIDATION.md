---
phase: 33
slug: fixture-metadata-and-contract-lock
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-02
---

# Phase 33 — Validation Strategy

## Quick Feedback Lane

- `rg -n "LFM2.5-1.2B-Thinking-Q4_K_M.gguf|architecture=lfm2|128000" tests/models/README.md src/emel/model/data.cpp`
- `rg -n "roles=system,user|tools=none|keep_past_thinking=false|add_generation_prompt=true" tools/generation_formatter_contract.hpp docs/benchmarks.md`

## Full Verification

- `scripts/quality_gates.sh`

## Notes

- Phase 33 validation is artifact-oriented because the phase closed fixture, metadata, and contract
  truth rather than a new runtime path.
- The maintained Liquid fixture provenance, metadata truth, and explicit contract are all present
  on repo-visible surfaces and are now backed by summary and verification artifacts.
