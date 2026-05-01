---
phase: 174
slug: sml-orchestration-surface-proof
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 174 — Validation Strategy

## Quick Feedback Lane

- `EMEL_ZIG_TEST_SHARDS=sm scripts/build_with_zig.sh`
- `ctest --test-dir build/zig -R '^emel_tests_sm$' --output-on-failure`

## Rule Compliance Evidence

- New transition rows use destination-first form.
- New required event payload fields use references.
- New actions are bounded and contain no runtime branching.
- SML state inspection is used for machine assertions.

## Notes

No unresolved escalations or manual-only blockers remain for this phase.

