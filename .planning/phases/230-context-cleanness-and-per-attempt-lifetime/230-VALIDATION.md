---
phase: 230
slug: context-cleanness-and-per-attempt-lifetime
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 230 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; quality gate scripts |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R emel_tests_io` |
| Full suite command | `EMEL_QUALITY_GATES_CHANGED_FILES=... scripts/quality_gates.sh` |
| Estimated runtime | focused lane under 30 seconds |

## Sampling Rate

- After context, runtime, or callback payload edits: run the focused I/O CTest lane.
- Before phase sign-off: run the scoped quality gate recorded in `230-VERIFICATION.md`.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 230-01-01 | 01 | 1 | STG-07 | dispatch-local storage | `staged_read::context` is empty and no request payload is retained in context. | doctest + static assert | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 230-01-02 | 01 | 1 | LIFE-02 | per-attempt lifetime | Per-attempt data stays on the same-RTC event stack; no staged-owned OS handle persists. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 230-01-03 | 01 | 1 | SNR-01 | residency ownership | Staged actor publishes caller-owned target only and never claims tensor residency. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |

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
