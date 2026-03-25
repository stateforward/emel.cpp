---
phase: 05
slug: gate-hardening
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-08
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest via CTest |
| **Config file** | `tools/paritychecker/CMakeLists.txt` |
| **Quick run command** | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~240 seconds |

---

## Sampling Rate

- **After every task commit:** Run `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- **After every plan wave:** Run `scripts/quality_gates.sh`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 240 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | VER-02 | subprocess regression | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ | ⬜ pending |
| 05-01-02 | 01 | 1 | VER-02 | gate smoke | `scripts/paritychecker.sh` | ✅ | ⬜ pending |
| 05-02-01 | 02 | 2 | VER-02 | planning-doc verification | `rg -n "VER-01|VER-02|Phase 5|subprocess|failure-path|manual-only" .planning/ROADMAP.md .planning/REQUIREMENTS.md` | ✅ | ⬜ pending |
| 05-02-02 | 02 | 2 | VER-02 | full repo gate | `scripts/quality_gates.sh` | ✅ | ⬜ pending |

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
- [x] Feedback latency < 240s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
