---
phase: 15
slug: runtime-adoption-and-proof
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-22
---

# Phase 15 - Validation Strategy

> Per-phase validation contract for runtime adoption, negative behavior, and parity proof.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest + paritychecker tool tests |
| **Config file** | `tests/generator/lifecycle_tests.cpp`, `tools/paritychecker/paritychecker_tests.cpp` |
| **Quick run command** | `./build/zig/emel_tests_bin --test-case='generator_*flash*' --no-breaks` |
| **Focused run command** | `ctest --test-dir build/zig --output-on-failure -R paritychecker_tests` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~900 seconds |

---

## Sampling Rate

- **After every task change:** run the focused generator flash doctests
- **After parity surface changes:** run `ctest --test-dir build/zig --output-on-failure -R paritychecker_tests`
- **Before phase closeout:** run `scripts/quality_gates.sh`
- **Max feedback latency:** 900 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | ARCH-01 | failing-first runtime observability proof | `./build/zig/emel_tests_bin --test-case='*generator*flash*' --no-breaks` | ✅ | ⬜ pending |
| 15-01-02 | 01 | 1 | ARCH-01, VER-02 | wrapper observability implementation | `./build/zig/emel_tests_bin --test-case='*generator*flash*' --no-breaks` | ✅ | ⬜ pending |
| 15-02-01 | 02 | 2 | PORT-03, VER-02 | negative runtime proof | `./build/zig/emel_tests_bin --test-case='*generator*flash*' --no-breaks` | ✅ | ⬜ pending |
| 15-03-01 | 03 | 2 | PAR-03 | parity publication proof | `ctest --test-dir build/zig --output-on-failure -R paritychecker_tests` | ✅ | ⬜ pending |
| 15-03-02 | 03 | 2 | PAR-03, VER-02 | regression closure | `scripts/quality_gates.sh` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠ flaky*

---

## Wave 0 Requirements

Existing runtime and parity test surfaces already exist. No extra Wave 0 scaffolding is required.

---

## Manual-Only Verifications

All planned Phase 15 behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have an `<automated>` verification command
- [x] Sampling continuity is preserved across all three plans
- [x] No manual-only or watch-mode verification is required
- [x] Feedback latency stays within the phase budget
- [x] `nyquist_compliant: true` is set in frontmatter

**Approval:** pending
