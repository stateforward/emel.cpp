---
phase: 03
slug: generator-initialization-wiring
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-08
---

# Phase 03 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest + CTest |
| **Config file** | `tools/paritychecker/CMakeLists.txt` |
| **Quick run command** | `scripts/paritychecker.sh` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~900 seconds |

---

## Sampling Rate

- **After every task commit:** Run `scripts/paritychecker.sh`
- **After every plan wave:** Run `scripts/quality_gates.sh`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 900 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | INIT-01 | unit/integration | `build/zig/emel_tests_bin --dt-test-case="*generator*"` | ✅ | ⬜ pending |
| 03-01-02 | 01 | 1 | INIT-01 | smoke | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello"` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 2 | INIT-02 | behavior | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello" | rg "initialize"` | ✅ | ⬜ pending |
| 03-02-02 | 02 | 2 | INIT-02 | unit/behavior | `build/zig/emel_tests_bin --dt-test-case="*generator*"` | ✅ | ⬜ pending |
| 03-02-03 | 02 | 2 | INIT-01, INIT-02 | repo-gate | `scripts/quality_gates.sh` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No new external service or fixture bootstrap
is required beyond the checked-in model catalog under `tests/models/` and the existing Zig build
targets.

---

## Manual-Only Verifications

All required phase behaviors should be machine-verifiable through the generator unit tests, the
paritychecker CLI smoke path, and the standard repo gates.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 900s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
