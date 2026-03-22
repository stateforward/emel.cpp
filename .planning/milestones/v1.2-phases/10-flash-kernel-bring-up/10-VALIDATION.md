---
phase: 10
slug: flash-kernel-bring-up
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-21
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest + CTest kernel-focused suites |
| **Config file** | `CMakeLists.txt`, `tests/kernel/lifecycle_tests.cpp`, `tests/kernel/aarch64_tests.cpp` |
| **Quick run command** | `ctest --output-on-failure -R kernel` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~600 seconds |

---

## Sampling Rate

- **After every task commit:** Run `ctest --output-on-failure -R kernel`
- **After every plan wave:** Run `scripts/quality_gates.sh`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 600 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | FLASH-01 | repro-test | `ctest --output-on-failure -R kernel` | ✅ | ⬜ pending |
| 10-01-02 | 01 | 1 | FLASH-01 | kernel-correctness | `ctest --output-on-failure -R kernel` | ✅ | ⬜ pending |
| 10-02-01 | 02 | 2 | FLASH-01 | backend-routing | `ctest --output-on-failure -R kernel` | ✅ | ⬜ pending |
| 10-02-02 | 02 | 2 | FLASH-02 | reuse-proof | `ctest --output-on-failure -R kernel && scripts/quality_gates.sh` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 600s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
