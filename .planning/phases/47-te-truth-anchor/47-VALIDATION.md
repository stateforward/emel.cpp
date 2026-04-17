---
phase: 47
slug: te-truth-anchor
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-14
---

# Phase 47 — Validation Strategy

## Quick Feedback Lane

- `shasum -a 256 tests/models/TE-75M-q8_0.gguf`
- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --no-breaks --source-file='*tests/model/*'`

## Full Verification

- `./build/zig/emel_tests_bin --no-breaks --source-file='*tests/model/*'`
- `scripts/quality_gates.sh`

## Notes

- The maintained TE fixture checksum and manifest regression checks passed during Phase 47
  verification.
- Later milestone proof work reran the repo gates green, so the truth-anchor surface remains
  validated at closeout time.
