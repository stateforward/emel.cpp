---
phase: 08
slug: generation-compare-output
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-10
---

# Phase 08 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | existing `bench_runner` + shell validation |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare` |
| **Full suite command** | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` |
| **Estimated runtime** | ~360 seconds |

---

## Sampling Rate

- **After every task commit:** Run `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare`
- **After every plan wave:** Run `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 360 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | COMP-01 | build | `cmake --build build/bench_tools_ninja --parallel --target bench_runner` | ✅ | ⬜ pending |
| 08-01-02 | 01 | 1 | COMP-01, COMP-02 | behavior | `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare | rg "^generation/"` | ✅ | ⬜ pending |
| 08-02-01 | 02 | 2 | COMP-02 | compare-smoke | `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` | ✅ | ⬜ pending |
| 08-02-02 | 02 | 2 | COMP-01, COMP-02 | audit-smoke | `EMEL_BENCH_CASE_INDEX=7 EMEL_BENCH_AUDIT_GENERATION_SEAMS=1 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare` | ✅ | ⬜ pending |

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
- [x] Feedback latency < 360s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
