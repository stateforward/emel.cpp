---
phase: 229
slug: staged-copy-progress-and-completion-semantics
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 229 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; quality gate scripts |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R emel_tests_io` |
| Full suite command | `EMEL_QUALITY_GATES_CHANGED_FILES=... scripts/quality_gates.sh` |
| Estimated runtime | focused lane under 30 seconds; scoped gate recorded in verification |

## Sampling Rate

- After copy or completion edits: run `emel_tests_io`.
- Before phase sign-off: run the scoped quality gate over staged-read source and lifecycle tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 229-01-01 | 01 | 1 | STG-04 | copy correctness | Accepted staged windows copy the intended source sub-span deterministically. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 229-01-02 | 01 | 1 | STG-05 | monotone progress | Non-divisible chunks cover the logical span without gaps or backtracking. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 229-01-03 | 01 | 1 | STG-06 | terminal completion | Exactly one success callback corresponds to full-span completion. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |

## Wave 0 Requirements

Existing staged-read lifecycle tests cover all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by focused CTest plus scoped quality gate.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
