---
phase: 234
slug: public-dispatch-tests
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 234 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest |
| Config file | `CMakeLists.txt` |
| Quick run command | `./build/emel_tests_bin --test-case="*staged-read dispatch*" --no-breaks` |
| Full suite command | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` |
| Estimated runtime | focused dispatch tests under 30 seconds |

## Sampling Rate

- After dispatch-surface edits: run the staged-read dispatch doctest filter.
- Before phase sign-off: run focused model and I/O CTest lanes.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 234-01-01 | 01 | 1 | TST-01 | success path dispatch | Public dispatch proves a fully successful staged-read path with observable SML state. | doctest | `./build/emel_tests_bin --test-case="*staged-read dispatch*" --no-breaks` | yes | green |
| 234-01-02 | 01 | 1 | TST-02 | failure path dispatch | Public dispatch proves representative guard failure through modeled error path. | doctest | `./build/emel_tests_bin --test-case="*staged-read dispatch*" --no-breaks` | yes | green |

## Wave 0 Requirements

Existing public dispatch doctests cover all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by targeted dispatch tests and focused CTest lanes.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
