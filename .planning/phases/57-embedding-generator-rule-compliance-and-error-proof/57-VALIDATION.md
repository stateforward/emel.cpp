---
phase: 57
slug: embedding-generator-rule-compliance-and-error-proof
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-16
---

# Phase 57 — Validation Strategy

## Quick Feedback Lane

- `./build/coverage/emel_tests_bin --no-breaks --test-case='embeddings generator action/guard contract stays explicit,*missing a required modality family*'`

## Full Verification

- `ctest --test-dir build/coverage -R '^emel_tests' -j 1`
- `EMEL_COVERAGE_CLEAN=1 scripts/test_with_coverage.sh`
- `scripts/paritychecker.sh`
- `scripts/fuzz_smoke.sh`
- `scripts/generate_docs.sh`

## Notes

- This phase is valid when the embedding-generator path keeps outcome routing in guards and
  transitions, and contract drift still locks to the maintained `model_invalid` error class.
