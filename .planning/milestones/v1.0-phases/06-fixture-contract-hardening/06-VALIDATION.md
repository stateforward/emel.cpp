---
phase: 06
slug: fixture-contract-hardening
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-08
---

# Phase 06 — Validation Strategy

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
- **After every plan wave:** Run `scripts/paritychecker.sh`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 240 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | HARN-02 | subprocess regression | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ | ✅ green |
| 06-01-02 | 01 | 1 | HARN-02 | behavior | `build/paritychecker_zig/paritychecker --help` | ✅ | ✅ green |
| 06-02-01 | 02 | 2 | HARN-02 | smoke | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1` | ✅ | ✅ green |
| 06-02-02 | 02 | 2 | HARN-02 | repo-gate | `scripts/paritychecker.sh` | ✅ | ✅ green |
| 06-02-03 | 02 | 2 | HARN-02 | full gate | `scripts/quality_gates.sh` | ✅ | ✅ green |

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

**Approval:** approved
