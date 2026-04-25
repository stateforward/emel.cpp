---
phase: 82
slug: diarization-request-and-audio-frontend-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-22
---

# Phase 82 Validation: Diarization Request And Audio Frontend Contract

**Validated:** 2026-04-22

## Automated Checks

| Check | Result | Notes |
|-------|--------|-------|
| `cmake --build build/coverage --target emel_tests_bin -j 8` | PASS | Native coverage build compiled the new diarization request tests. |
| `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1` | PASS | Focused diarization shard passed. |
| `EMEL_COVERAGE_CHANGED_ONLY=1 scripts/test_with_coverage.sh` | PASS | Changed-file coverage passed at `97.7%` line and `60.0%` branch. |
| `scripts/quality_gates.sh` | PASS | Passed; benchmark snapshot warnings were tolerated by the gate. |
| `git diff --check` | PASS | No whitespace errors. |

## Requirement Coverage

| Requirement | Evidence | Status |
|-------------|----------|--------|
| DIA-01 | `event::prepare` accepts only in-memory `std::span<const float>` PCM with explicit sample rate and channel count. Guards reject non-16 kHz and non-mono requests. | Covered |
| DIA-02 | `encoder::feature_extractor::detail::compute` derives the maintained caller-owned feature matrix natively after validation without allocation or external runtime fallback. | Covered |
| DIA-03 | Guards and tests cover invalid profile, sample rate, channel count, PCM shape, and feature-output capacity with explicit errors. | Covered |

## Residual Risk

Phase 82 does not execute Sortformer tensors or produce diarization probabilities. The feature-extractor
feature contract is deterministic and native, but full runtime equivalence belongs to Phase 83 and
later parity proof belongs to Phase 85.
