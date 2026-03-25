---
phase: 01
slug: generation-harness-contract
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-07
---

# Phase 01 — Validation Strategy

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
| 01-01-01 | 01 | 1 | HARN-01 | integration | `scripts/paritychecker.sh` | ✅ | ⬜ pending |
| 01-01-02 | 01 | 1 | HARN-01 | smoke | `build/paritychecker_zig/paritychecker --help` | ✅ | ⬜ pending |
| 01-01-03 | 01 | 1 | HARN-01 | negative | `! build/paritychecker_zig/paritychecker --generation --text "hello"` | ✅ | ⬜ pending |
| 01-02-01 | 02 | 2 | HARN-01 | integration | `scripts/paritychecker.sh` | ✅ | ⬜ pending |
| 01-02-02 | 02 | 2 | HARN-02 | negative | `! build/paritychecker_zig/paritychecker --generation --model tests/models/distilgpt2.Q2_K.gguf --text "hello"` | ✅ | ⬜ pending |
| 01-02-03 | 02 | 2 | HARN-02 | smoke | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello"` | ✅ | ⬜ pending |
| 01-02-04 | 02 | 2 | HARN-02 | repo-gate | `scripts/quality_gates.sh` | ✅ | ⬜ pending |

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
- [x] Feedback latency < 900s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
