---
phase: 235
slug: scope-and-non-regression-guardrails
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 235 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` |
| Full suite command | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` |
| Estimated runtime | focused lanes under 1 minute; full gate recorded at closeout |

## Sampling Rate

- After guardrail edits: run the exact guardrail doctest filters recorded in `235-VERIFICATION.md`.
- Before milestone sign-off: run the full quality gate.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 235-01-01 | 01 | 1 | GRD-01 | loader ownership leak | Guardrail fails if staged-read file syscall ownership leaks into `model/loader`. | doctest/source scan | `./build/emel_tests_bin --test-case="model loader io boundary uses actor events without helper exposure" --no-intro` | yes | green |
| 235-01-02 | 01 | 1 | GRD-02 | tensor residency leak | Guardrail fails if staged-load residency moves out of `model/tensor`. | doctest/source scan | `./build/emel_tests_bin --test-case="model_tensor_owns_staged_read_residency_boundary" --no-intro` | yes | green |
| 235-01-03 | 01 | 1 | GRD-03 | coroutine scope creep | Guardrail fails if staged scheduling/coroutine scaffolding appears without approval. | doctest/source scan | `./build/emel_tests_bin --test-case="phase 235 grd-03 staged scheduling has no coroutine scaffolding tokens" --no-intro` | yes | green |
| 235-01-04 | 01 | 1 | GRD-04 | mmap regression | Existing mmap success/release semantics remain green. | doctest | `./build/emel_tests_bin --test-case="io mmap returns a deterministic mapped descriptor on success,io mmap release happy path returns slot to the free pool" --no-intro` | yes | green |
| 235-01-05 | 01 | 1 | GRD-05 | bulk read regression | Existing bulk `io/read` route remains green. | doctest | `./build/emel_tests_bin --test-case="io loader read copy batch routes once through io read" --no-intro` | yes | green |

## Wave 0 Requirements

Existing guardrail and non-regression doctests cover all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated or source-scan verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by focused doctest filters and CTest lanes.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
