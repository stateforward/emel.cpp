---
phase: 11
slug: generator-flash-adoption
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-21
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest + CTest generator-focused suites |
| **Config file** | `CMakeLists.txt`, `tests/generator/lifecycle_tests.cpp`, `tests/generator/detail_tests.cpp` |
| **Quick run command** | `ctest --output-on-failure -R generator` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~600 seconds |

---

## Sampling Rate

- **After every task commit:** Run `ctest --output-on-failure -R generator`
- **After every plan wave:** Run `scripts/quality_gates.sh`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 600 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | GEN-01 | runtime-routing | `ctest --output-on-failure -R generator` | ✅ | ⬜ pending |
| 11-01-02 | 01 | 1 | GEN-01 | observability | `ctest --output-on-failure -R generator` | ✅ | ⬜ pending |
| 11-02-01 | 02 | 2 | GEN-02 | deterministic-failure | `ctest --output-on-failure -R generator` | ✅ | ⬜ pending |
| 11-02-02 | 02 | 2 | GEN-01, GEN-02 | regression-gate | `ctest --output-on-failure -R generator && scripts/quality_gates.sh` | ✅ | ⬜ pending |

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
