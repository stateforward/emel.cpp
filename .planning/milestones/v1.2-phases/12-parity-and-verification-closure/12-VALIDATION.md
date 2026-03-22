---
phase: 12
slug: parity-and-verification-closure
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-21
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest + paritychecker subprocess tests |
| **Config file** | `tools/paritychecker/CMakeLists.txt`, `tools/paritychecker/paritychecker_tests.cpp` |
| **Quick run command** | `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~900 seconds |

---

## Sampling Rate

- **After every task commit:** `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests`
- **After every plan wave:** `scripts/quality_gates.sh`
- **Before `$gsd-verify-work`:** the paritychecker subprocess suite and full quality gates must be green
- **Max feedback latency:** 900 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | PAR-01, PAR-02 | source-selection | `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests` | ✅ | ✅ green |
| 12-01-02 | 01 | 1 | PAR-01 | proof-surface | `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1` | ✅ | ✅ green |
| 12-02-01 | 02 | 2 | PAR-02 | bounded-long | `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 8` | ✅ | ✅ green |
| 12-02-02 | 02 | 2 | VER-01 | regression-gate | `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests && scripts/quality_gates.sh` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠ flaky*

---

## Wave 0 Requirements

Existing Phase 10 and 11 verification already covers kernel correctness, generator adoption, and
negative flash-selection behavior. Phase 12 only needs to add paritychecker-level proof-surface and
bounded-long decode coverage.

---

## Manual-Only Verifications

All required Phase 12 behaviors are automatable.

---

## Validation Sign-Off

- [x] All tasks have automated verification
- [x] Existing prior-phase verification covers kernel/generator truths Phase 12 builds on
- [x] No watch-mode or interactive-only validation paths
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** passed
