---
phase: 54
slug: omniembed-execution-contract-runtime-cutover
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-14
---

# Phase 54 — Validation Strategy

## Quick Feedback Lane

- `./build/audit-native/emel_tests_bin --no-breaks --test-case='*missing a required modality family*'`

## Full Verification

- `EMEL_COVERAGE_BUILD_DIR=build/coverage-phase54 EMEL_COVERAGE_CLEAN=1 ./scripts/test_with_coverage.sh`
- `EMEL_COVERAGE_BUILD_DIR=build/coverage-phase54 ./scripts/quality_gates.sh`

## Notes

- The native audit build was used only for the targeted regression because the clean,
  repo-truthful full verification was the fresh `build/coverage-phase54` coverage run.
- The new regression proves initialization now rejects a broken required modality family before the
  maintained TE generator session can start.
- The coverage run required isolating `tests/sm/*` into a dedicated CTest shard so the generator
  and embeddings slice could execute without cross-suite doctest-process contamination.
