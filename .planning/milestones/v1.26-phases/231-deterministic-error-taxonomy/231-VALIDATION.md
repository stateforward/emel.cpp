---
phase: 231
slug: deterministic-error-taxonomy
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 231 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; quality gate scripts |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R emel_tests_io` |
| Full suite command | `EMEL_QUALITY_GATES_CHANGED_FILES=... scripts/quality_gates.sh` |
| Estimated runtime | focused lane under 30 seconds |

## Sampling Rate

- After error, guard, or transition edits: run focused staged-read lifecycle tests.
- Before phase sign-off: run the scoped quality gate recorded in `231-VERIFICATION.md`.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 231-01-01 | 01 | 1 | ESG-01 | pre-I/O guard errors | Pre-I/O failures map to named deterministic staged-read categories. | doctest + source scan | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 231-01-02 | 01 | 1 | ESG-02A | source contract errors | Null, mismatched, and insufficient source spans map to named deterministic errors. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 231-01-03 | 01 | 1 | ESG-03 | stage contract errors | Sequencing and stage-contract failures remain explicit and deterministic. | doctest | `ctest --test-dir build --output-on-failure -R emel_tests_io` | yes | green |
| 231-01-04 | 01 | 1 | ESG-04 | exception boundary | Staged-read boundary remains exception-free across public actor dispatch. | compile + source scan | `ninja -C build emel_tests_bin` | yes | green |

## Deferred Requirements

- `ESG-02B` is explicitly deferred/future and not counted as an in-scope v1.26 blocker because
  file-backed staged-read ownership is not approved in this milestone.

## Wave 0 Requirements

Existing staged-read lifecycle tests cover all in-scope phase requirements.

## Manual-Only Verifications

All in-scope phase behaviors have automated or source-scan verification.

## Validation Sign-Off

- [x] All in-scope tasks have automated verification.
- [x] Sampling continuity is preserved by focused CTest plus scoped quality gate.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
