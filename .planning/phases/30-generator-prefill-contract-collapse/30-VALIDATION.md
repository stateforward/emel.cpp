---
phase: 30
slug: generator-prefill-contract-collapse
status: ready
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 30 — Validation Strategy

## Quick Feedback Lane

- `ctest --test-dir build/zig --output-on-failure -R generator_lifecycle_tests`

## Full Verification

- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh`

## Notes

- Phase 30 is an architecture-preserving refactor. Maintained parity and bench proof are deferred
  to Phase 32, but generator topology and docs must stay aligned as the state table changes.
