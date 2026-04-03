---
phase: 34-initializer-surface-shrink-and-proof
verified: 2026-04-02T22:20:00Z
status: passed
score: 3/3 phase truths verified
---

# Phase 34 Verification Report

**Phase Goal:** Prove the EMEL-owned Qwen3 E2E probe as a final executable on the maintained
workload without reference-assisted bootstrap.
**Verified:** 2026-04-02T22:20:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The maintained EMEL row builds as a final linked executable for the canonical Qwen3 slice. | ✓ VERIFIED | `./scripts/embedded_size.sh --json` produced the EMEL row at `build/embedded_size/emel_probe_build/emel_qwen3_e2e_probe` with runtime smoke passing on the maintained fixture. |
| 2 | The published EMEL probe no longer relies on fallback-vocab sentinel blobs that distort executable size. | ✓ VERIFIED | [events.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/encoders/events.hpp), [guards.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/encoders/guards.hpp), and [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/encoders/bpe/detail.hpp) now require a real vocab path instead of manufacturing empty `vocab{}` sentinels. |
| 3 | The maintained EMEL probe size dropped from the earlier distorted 56 MB image to the corrected ~4.07 MB executable while keeping the same E2E workload. | ✓ VERIFIED | The latest `embedded_size.sh --json` run reports `emel.raw_bytes = 4073016`, `section_bytes = 1323877`, and `runtime_smoke = passed`. |

**Score:** 3/3 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tools/embedded_size/emel_probe/main.cpp` | Final EMEL runner executable path | ✓ EXISTS + SUBSTANTIVE | Maintained Qwen3 E2E probe entrypoint exists and is exercised by the size harness. |
| `scripts/embedded_size.sh` | Executable-size build and smoke driver | ✓ EXISTS + SUBSTANTIVE | Builds the EMEL and reference rows and reports raw/stripped/section sizes plus smoke metadata. |
| `src/emel/text/encoders/events.hpp` | No giant fallback vocab sentinel | ✓ EXISTS + FIXED | `event::encode` now requires a real vocab reference. |
| `src/emel/text/unicode_data.cpp` | Out-of-line unicode data | ✓ EXISTS + ADDITIVE | Large unicode data moved out of headers into compiled storage. |

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| E2E-01 | ✓ PHASE-ALIGNED | Claimed formally in Phase 38 traceability backfill |
| E2E-02 | ✓ PHASE-ALIGNED | Claimed formally in Phase 38 traceability backfill |

## Automated Checks

- `cmake --build build/debug --target emel_tests_bin`
- `./build/debug/emel_tests_bin --test-case='*encoder*'`
- `./build/debug/emel_tests_bin --test-case='*tokenizer*'`
- `./scripts/embedded_size.sh --json`

## Verification Notes

- This phase verifies the EMEL row and harness correctness. Publication freshness is deferred to
  the closeout phase.

---
*Verified: 2026-04-02T22:20:00Z*
*Verifier: the agent*
