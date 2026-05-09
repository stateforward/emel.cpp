---
phase: 227
slug: staged-read-strategy-component-boundary
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 227 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; quality gate scripts |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R emel_tests_io` |
| Full suite command | `EMEL_COVERAGE_CHANGED_ONLY=0 scripts/test_with_coverage.sh` |
| Estimated runtime | ~3 minutes for full coverage evidence |

## Sampling Rate

- After staged-read scaffold edits: run `ninja -C build emel_tests_bin` and the focused I/O CTest lane.
- Before phase sign-off: run full coverage when scoped changed-file instrumentation cannot measure
  header-only work.
- Max feedback latency target: focused lane under 30 seconds.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 227-01-01 | 01 | 1 | STG-01 | RC-SML-RTC-NO-QUEUE | Dedicated `io/staged_read` actor exists with canonical machine alias and no mmap/device/coroutine scope creep. | doctest + source scan | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 227-01-02 | 01 | 1 | STG-01 | coverage | Supplemental full coverage handles header-only scoped instrumentation gap. | coverage gate | `EMEL_COVERAGE_CHANGED_ONLY=0 scripts/test_with_coverage.sh` | yes | green |

## Wave 0 Requirements

Existing doctest and CTest infrastructure covers this phase.

## Manual-Only Verifications

All phase behaviors have automated or source-scan verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by focused CTest plus supplemental coverage.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
