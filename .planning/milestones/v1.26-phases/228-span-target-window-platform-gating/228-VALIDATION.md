---
phase: 228
slug: span-target-window-platform-gating
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 228 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R emel_tests_io` |
| Full suite command | `scripts/quality_gates.sh` when scoped by changed files |
| Estimated runtime | focused lane under 30 seconds |

## Sampling Rate

- After guard or transition edits: run the focused I/O CTest lane.
- Before phase sign-off: verify all guard rejection and accepted-path cases named in
  `228-VERIFICATION.md`.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 228-01-01 | 01 | 1 | STG-02 | invalid source/stage contract | Invalid logical span, chunk size, and offset overflow reject before staged work. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 228-01-02 | 01 | 1 | STG-03 | invalid target window | Null or undersized target windows reject before staged work. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 228-01-03 | 01 | 1 | PLAT-02 | unsupported host/resource | Platform support is guarded by explicit transition/source predicate. | source scan + compile | `ninja -C build emel_tests_bin` | yes | green |

## Wave 0 Requirements

Existing staged-read lifecycle tests cover all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated or source-scan verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by focused I/O tests.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
