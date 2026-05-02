---
phase: 169
slug: sml-orchestration-behavior-preservation
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 169 — Validation Strategy

## Quick Feedback Lane

- `EMEL_ZIG_TEST_SHARDS=sm scripts/build_with_zig.sh`
- `ctest --test-dir build/zig -R '^emel_tests_sm$' --output-on-failure`

## Rule Compliance Evidence

- Phase 174 contains the active test proof and rule-compliance notes.

