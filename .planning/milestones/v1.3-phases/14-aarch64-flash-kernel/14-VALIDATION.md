---
phase: 14
slug: aarch64-flash-kernel
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
---

# Phase 14 - Validation Strategy

> Per-phase validation contract for execution feedback and requirement sampling.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest + CTest kernel-focused suites |
| **Config file** | `CMakeLists.txt`, `tests/kernel/aarch64_tests.cpp`, `tests/kernel/lifecycle_tests.cpp` |
| **Quick run command** | `ctest --output-on-failure -R kernel` |
| **Focused run command** | `./build/debug/emel_tests_bin --test-case='*aarch64*flash*' --no-breaks --force-colors=0` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~600 seconds |

---

## Sampling Rate

- **After every task commit:** Run `ctest --output-on-failure -R kernel`
- **After each plan:** Run the focused flash kernel doctests plus `ctest --output-on-failure -R kernel`
- **Before phase closeout:** Run `scripts/quality_gates.sh`
- **Max feedback latency:** 600 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | PORT-01 | failing-first seam repro | `ctest --output-on-failure -R kernel` | ✅ | ⬜ pending |
| 14-01-02 | 01 | 1 | PORT-01, PORT-02 | optimized backend implementation | `ctest --output-on-failure -R kernel` | ✅ | ⬜ pending |
| 14-02-01 | 02 | 2 | PORT-02 | proof harness and allocation guard reuse | `ctest --output-on-failure -R kernel` | ✅ | ⬜ pending |
| 14-02-02 | 02 | 2 | PORT-01, PORT-02 | correctness, reuse, and alloc-free proof | `ctest --output-on-failure -R kernel && scripts/quality_gates.sh` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠ flaky*

---

## Wave 0 Requirements

Existing test infrastructure covers the phase. No extra Wave 0 scaffold is required before
execution.

---

## Manual-Only Verifications

All planned Phase 14 behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have an `<automated>` verification command
- [x] Sampling continuity is preserved across both plans
- [x] No watch-mode or manual-only verification is required
- [x] Feedback latency stays under the phase budget
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** pending
