---
phase: 04
slug: deterministic-generation-parity
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-08
---

# Phase 04 - Validation Strategy

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
| 04-01-01 | 01 | 1 | GEN-01, GEN-02 | unit/integration | `build/zig/emel_tests_bin --dt-test-case="*generator*"` | ✅ | ⬜ pending |
| 04-01-02 | 01 | 1 | GEN-01, GEN-02 | smoke | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello" --max-tokens 1` | ✅ | ⬜ pending |
| 04-02-01 | 02 | 2 | PARI-01 | behavior | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello" --max-tokens 1 | rg "parity"` | ✅ | ⬜ pending |
| 04-02-02 | 02 | 2 | PARI-01 | repo-gate | `scripts/paritychecker.sh` | ✅ | ⬜ pending |
| 04-03-01 | 03 | 3 | PARI-02 | behavior | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello" --max-tokens 1 --dump` | ✅ | ⬜ pending |
| 04-03-02 | 03 | 3 | GEN-01, GEN-02, PARI-01, PARI-02 | coverage | `scripts/test_with_coverage.sh` | ✅ | ⬜ pending |
| 04-03-03 | 03 | 3 | GEN-01, GEN-02, PARI-01, PARI-02 | repo-gate | `scripts/quality_gates.sh` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers the phase. The required fixture, paritychecker binary target,
generator unit tests, and repo gates already exist.

---

## Manual-Only Verifications

Manual inspection is limited to confirming that the generation output now reports parity evidence
instead of the Phase 3 initialize-only message. All required success criteria should still be
machine-verifiable through the listed commands.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 900s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
