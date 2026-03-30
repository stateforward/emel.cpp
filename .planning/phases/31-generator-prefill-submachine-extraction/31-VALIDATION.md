---
phase: 31
slug: generator-prefill-submachine-extraction
status: ready
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 31 — Validation Strategy

## Quick Feedback Lane

- `./scripts/build_with_zig.sh`
- `build/zig/emel_tests_bin --test-case='*generator_prefill*'`
- `build/zig/emel_tests_bin --test-case='*generator_sm_*'`

## Full Verification

- `build/zig/emel_tests_bin --test-case='*qwen3*generator*'`
- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh`

## Notes

- Phase 31 is the structural extraction cut. Maintained parity/bench proof is still closed in
  Phase 32, but the parent/child boundary must already preserve the existing generator behavior.
